/*
 * main.cpp
 *
 *  Created on: Aug 20, 2014
 *      Author: dimalit
 */

#include <model_e3.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
using namespace pb;

#include "solver.h"

#include <petscts.h>
#include <mpi.h>
#include <cstring>
#include <unistd.h>

E3Config pconfig;
E3PetscSolverConfig sconfig;
E3State state;

int rank;
int size;
clock_t t1;		// for time counting in step_func
int max_steps; double max_time;
bool use_step = false;

void vec_to_state(Vec v, E3State*);
void state_to_vec(const E3State* state, Vec v);
void step_func(Vec u, Vec rhs, int steps, double time);

// TMP
//#include <fcntl.h>

void broadcast_message(google::protobuf::Message& msg){
	char* buf; int buf_size;

	if(rank == 0)
		buf_size = msg.ByteSize();
	MPI_Bcast(&buf_size, 1, MPI_INT, 0, PETSC_COMM_WORLD);

	buf = new char[buf_size];

	if(rank == 0)
		msg.SerializeToArray(buf, buf_size);

	MPI_Bcast(buf, buf_size, MPI_BYTE, 0, PETSC_COMM_WORLD);

	if(rank != 0)
		msg.ParseFromArray(buf, buf_size);

	delete[] buf;
}

void parse_with_prefix(google::protobuf::Message& msg, int fd){
	int size;
	int ok = read(fd, &size, sizeof(size));
	assert(ok == sizeof(size));

	//TODO:without buffer cannot read later bytes
	char *buf = (char*)malloc(size);
	int read_size = 0;
	while(read_size != size){
		ok = read(fd, buf+read_size, size-read_size);
		read_size+=ok;
		assert(ok > 0 || read_size==size);
	}
	msg.ParseFromArray(buf, size);
	free(buf);
}

int main(int argc, char** argv){

//	int go = 0;
//	while(go==0){
//		sleep(1);
//	}

//	close(0);
//	open("../ode-env/all.tmp", O_RDONLY);

	if(argc > 1 && strcmp(argv[1], "use_step")==0){
		use_step = true;
		argc--;
		argv++;
	}

	PetscErrorCode ierr;
	ierr = PetscInitialize(&argc, &argv, (char*)0, (char*)0);CHKERRQ(ierr);

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);

	if(rank==0){
		E3Model all;
		//all.ParseFromFileDescriptor(0);
		parse_with_prefix(all, 0);

		sconfig.CopyFrom(all.sconfig());
		pconfig.CopyFrom(all.pconfig());
		state.CopyFrom(all.state());

		a0 = state.a0();
	}
	MPI_Bcast(&a0, 1, MPI_INT, 0, PETSC_COMM_WORLD);

	broadcast_message(sconfig);
	broadcast_message(pconfig);

	// set global parameters
	init_step = sconfig.init_step();
	atolerance = sconfig.atol();
	rtolerance = sconfig.rtol();
	m = pconfig.m();
	n = pconfig.n();
	theta_e = pconfig.theta_e();
	delta_e = pconfig.delta_e();
	r_e = pconfig.r_e();
	gamma_0_2 = pconfig.gamma_0_2();

	use_ifunction = false;
	use_ijacobian = false;
	if(sconfig.solver()!=sconfig.rhs)
		use_ifunction = true;
	if(sconfig.solver()==sconfig.ijacobian)
		use_ijacobian = true;

	Vec u;
	VecCreate(PETSC_COMM_WORLD, &u);
	VecSetType(u, VECMPI);

	int addition = rank==0 ? 2 : 0;
	VecSetSizes(u, addition+pconfig.m()*3/size, PETSC_DECIDE);

	state_to_vec(&state, u);

	if(rank == 0){
		int ok;
		ok = read(0, &max_steps, sizeof(max_steps));
			assert(ok==sizeof(max_steps));
		ok = read(0, &max_time, sizeof(max_time));
			assert(ok==sizeof(max_time));
	}

	MPI_Bcast(&max_steps, 1, MPI_INT, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&max_time, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	t1 = clock();

	if(sconfig.model() == "te")
		solve_te(u, max_steps, max_time, step_func);
	else if(sconfig.model() == "tm")
		solve_tm(u, max_steps, max_time, step_func);
	else{
		fprintf(stderr, "assert");
		fflush(stderr);
		assert(false);
	}

	VecDestroy(&u);

	ierr = PetscFinalize();CHKERRQ(ierr);
	return 0;
}

void step_func(Vec res, Vec res_rhs, int passed_steps, double passed_time){
	clock_t t2 = clock();
	double dtime = (double)(t2-t1)/CLOCKS_PER_SEC;

	if(!use_step && passed_steps < max_steps && passed_time < max_time)
		return;

	E3Solution sol;
	if(rank==0){
		for(int i=0; i<m; i++){
			sol.mutable_state()->add_particles();
			sol.mutable_d_state()->add_particles();
		}
	}

	vec_to_state(res, sol.mutable_state());
	vec_to_state(res_rhs, sol.mutable_d_state());
	sol.mutable_state()->set_a0(a0);
	sol.mutable_d_state()->set_a0(0);

	if(rank == 0){
		// 1 write time and steps
		write(1, &passed_steps, sizeof(passed_steps));
		write(1, &passed_time, sizeof(passed_time));

		// 2 write state
		int size = sol.ByteSize();
		write(1, &size, sizeof(size));
//		fprintf(stderr, "size=%d text:%s", size, sol.DebugString().c_str());
		sol.SerializeToFileDescriptor(1);
	}

	t1 = t2;
}

void state_to_vec(const E3State* state, Vec v){
	int vecsize;
	VecGetSize(v, &vecsize);
	assert(vecsize == pconfig.m()*3+2);

	double *arr;
	VecGetArray(v, &arr);

	PetscInt* borders;
	VecGetOwnershipRanges(v, (const PetscInt**)&borders);

	if(rank == 0){

		// write E and phi
		arr[0] = state->e();
		arr[1] = state->phi();

		for(int r = size-1; r>=0; r--){		// go down - because last will be mine
			int lo = borders[r];
			int hi = borders[r+1];
			if(r==0)
				lo += 2;

			assert((lo-2)%3 == 0);
			assert((hi-2)%3 == 0);

			int first = (lo-2) / 3;
			int count = (hi - lo) / 3;

			for(int i=0; i<count; i++){
				E3State::Particles p = state->particles(first+i);
				arr[2 + i*3+0] = p.eta();
				arr[2 + i*3+1] = p.a();
				arr[2 + i*3+2] = p.ksi();
			}

			if(r!=0)
				MPI_Send(arr+2, count*3, MPI_DOUBLE, r, 0, PETSC_COMM_WORLD);

		}// for

	}// if rank == 0
	else{
		int count3 = borders[rank+1] - borders[rank];
		MPI_Status s;
		int ierr = MPI_Recv(arr, count3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD, &s);
		assert(MPI_SUCCESS == ierr);
	}// if rank != 0

	VecRestoreArray(v, &arr);
}

void vec_to_state(Vec v, E3State* state){
	double *arr;
	VecGetArray(v, &arr);

	PetscInt* borders;
	VecGetOwnershipRanges(v, (const PetscInt**)&borders);

	if(rank == 0){

		PetscScalar* buf = (PetscScalar*)malloc(sizeof(PetscScalar)*(borders[1]-borders[0]));

		state->set_e(arr[0]);
		state->set_phi(arr[1]);

		for(int r = 0; r<size; r++){
			int lo = borders[r];
			int hi = borders[r+1];
			if(r==0)
				lo += 2;

			assert((lo-2)%3 == 0);
			assert((hi-2)%3 == 0);

			int first = (lo-2) / 3;
			int count = (hi - lo) / 3;

			MPI_Status s;
			if(r!=0){
				int ok = MPI_Recv(buf, count*3, MPI_DOUBLE, r, 0, PETSC_COMM_WORLD, &s);
				assert(MPI_SUCCESS == ok);
			}

			for(int i=0; i<count; i++){
				E3State::Particles* p = state->mutable_particles(first+i);
				p->set_eta(buf[2 + i*3+0]);
				p->set_a(buf[2 + i*3+1]);
				p->set_ksi(buf[2 + i*3+2]);
			}
		}// for

		free(buf);
	}// if rank == 0
	else{
		int count3 = borders[rank+1] - borders[rank];
		int ierr = MPI_Send(arr, count3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD);
		assert(MPI_SUCCESS == ierr);
	}// if rank != 0

	VecRestoreArray(v, &arr);
}

