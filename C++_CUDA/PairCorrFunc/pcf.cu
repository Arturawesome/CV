/*
 ============================================================================
Name       : pcf.cu
Author     : Nasyrov A.D.

Copyright  : Your copyright notice
Description:	calculation of pair corrleation fucntion in lquid system
				calcualtion radial diatribution fucntion
Еechnology : CUDA, Object Oriented Programming

The Titan X parametres:
Maximum number of threads per block:           1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
 ============================================================================
*/


#include <string.h>
#include <math.h>
#include <chrono>

#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <stdio.h>
#include<cuda_runtime.h>
using namespace std::chrono;

__global__ void helloFromGPU(void)
{

    printf("Hello from gpu \n");
}

extern "C" __global__ void returnOrderNeighboris(int *d_neigh_list_all, int *d_neigh_list_all_help, int sphere_order, int nparts, int num_max_neighbors, double *d_n_order)//, int order)
{
	// расчет относительно одной частицы
		//h_neigh_list_all_1[num_max_neighbors * (id_CENTER_part_in_array - 1) * sphere_order + counters[id_CENTER_part_in_array - 1]] = value;
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс уцентральной частицы (-1)
	int id_CENTER_part_in_array = id_part * num_max_neighbors * sphere_order;			//индекс центральной частицы в массиве данных

	bool flag = false;
	bool flag_1 = false;
	bool flag_2 = false;
	bool flag_3 = false;

 	for(int order = 1; order < sphere_order; order ++)
 	{
		int count = 0;
		for(int neigh = 0; d_neigh_list_all[id_CENTER_part_in_array + neigh + num_max_neighbors * (order - 1)]; neigh++)
		{ //neigh - порядок соседней частицы на пред сфере (1-я частица 2-я частьиц итл)

			int neigh_id = d_neigh_list_all[id_CENTER_part_in_array + neigh + num_max_neighbors * (order - 1)] - 1;
			//индекс соседа пред порядка (-1)
			//printf("order %i neigh_id %i \n", order, neigh_id);

			int neigh_id_in_array = neigh_id * num_max_neighbors * sphere_order;
			//индекс соседа в массиве (-1) (*)

			for(int neigh_1 = 0; d_neigh_list_all_help[neigh_id_in_array + neigh_1]; neigh_1 ++)
			{//neigh_1 - порядок соседней частицы на первой сфере (1-я частица 2-я частьиц итл)

				int el = d_neigh_list_all_help[neigh_id_in_array + neigh_1];
// 				//индекс соседей на первой сфере для частицы neigh_id_in_array (*)

				//ниже идет циклы с уловиями добавления ее в массив
				flag = false;
				flag_1 = false;
				flag_2 = false;
				flag_3 = false;
				// исключаем id с пред сферы
				for(int neigh_2 = 0; d_neigh_list_all[id_CENTER_part_in_array + neigh_2 + num_max_neighbors * (order - 1)]; neigh_2++)
				{
					int el_pred = d_neigh_list_all[id_CENTER_part_in_array + neigh_2 + num_max_neighbors * (order - 1)];
					if (el == el_pred)
					{
						flag = true;
						break;
					}
				}
				// исключаем повторы на рассчетной сфере
				for(int neigh_2 = 0; d_neigh_list_all[id_CENTER_part_in_array + neigh_2 + num_max_neighbors * (order)]; neigh_2++)
				{
					int el_pred = d_neigh_list_all[id_CENTER_part_in_array + neigh_2 + num_max_neighbors * (order)];
					if (el == el_pred)
					{
						flag_1 = true;
						break;
					}
				}
				// исклбчаем повтор с центральной частицей
				if(el == (id_part+1)) flag_2 = true;

				//исклбючаем повторы с пред-предыдушщей сферой
				if (order >= 2)
				{
					for(int neigh_3 = 0; d_neigh_list_all[id_CENTER_part_in_array + neigh_3 + num_max_neighbors * (order - 2)]; neigh_3++)
					{
						int el_pred = d_neigh_list_all[id_CENTER_part_in_array + neigh_3 + num_max_neighbors * (order - 2)];
						if (el == el_pred)
						{
							flag_3 = true;
							break;
						}
					}
				}
				if ((flag || flag_1 || flag_2 || flag_3) == false)
				{
 					 d_neigh_list_all[id_CENTER_part_in_array + num_max_neighbors * (order) + count] = el;
					 count++;
				}

			}
		}
		d_n_order[sphere_order*id_part + order] += double(count)/double(nparts);
 	}

}
extern "C" __global__ void check_data(double *d_xcord, double *d_ycord)
{
int order = threadIdx.x;
	printf("In CHECK \n order = %i \n", order);
	printf("x_center =  %d \n",  d_xcord[1]);


	for(int id_part = 0;id_part < 10; id_part ++)
	{	// id_part - the center particle
		printf("x_center =  %lf \n",  d_xcord[id_part]);
		printf("y_center =  %lf \n",  d_ycord[id_part]);


	}
}

extern "C" __global__ void calc_distance(double *d_pcf, int *d_neigh_list_all, double *d_xcord, double *d_ycord, double x_min, double x_max, double y_min, double y_max, double dr, int len_r,int nparts, int num_max_neighbors, int sphere_order, int iter, int id_iter)
{
	/*
	Нельзя делать рассчет относительно каждой частицы, т.к. будут происходить атомарные операции
	(несколько потоков записывают инфорамцию в одну ячейке памяти)

	Решение - делать рассчет относительно корреляционных сфер
	каждый поток будет записывать информацию только в ячейки определенной корреляционной сферы

	использовать колиество потоков равное кол-ву сфер
	*/
	int order = blockIdx.x;

	double x_center, y_center;
	double x_neigh, y_neigh, distance;
	double x_box = x_max - x_min;
	double y_box = y_max - y_min;
	int ind_pcf;
	double r_right, r_left, r_up, r_bottom;
	double norm_const =  nparts * 4 * M_PI * dr;
	int id_part = nparts * id_iter /iter;
	int lim_npart = nparts * (id_iter + 1) /iter;

	for(id_part; id_part < lim_npart; id_part ++)
	{	// id_part - the center particle
		x_center = d_xcord[id_part];
		y_center = d_ycord[id_part];
		r_right = x_center+x_box/2.1;
		r_left = x_center-x_box/2.1;
		r_up = y_center+y_box/2.1;
		r_bottom = y_center-y_box/2.1;
		//printf(" order = %i id_part = %i, x_center =  %f, y_center =  %f \n", order, id_part +1, d_xcord[id_part], d_ycord[id_part]);
		//.printf(" order = %i id_part = %i, y_center =  %f \n", order, id_part+1, d_ycord[id_part]);

		int id_CENTER_part_in_array = id_part * num_max_neighbors * sphere_order;			//индекс центральной частицы в массиве данных
		//printf("id_CENTER_part_in_array = %d \n", id_CENTER_part_in_array);

		for(int neigh = 0; d_neigh_list_all[id_CENTER_part_in_array + neigh + num_max_neighbors * order]; neigh++)
		{
			int neigh_id = d_neigh_list_all[id_CENTER_part_in_array + neigh + num_max_neighbors* order] - 1;
			//индекс соседа пред порядка (-1)
			//printf("neigh_id =  %f \n", neigh_id + 1);

			x_neigh = d_xcord[neigh_id];
			y_neigh = d_ycord[neigh_id];

			//printf(" order = %i neigh_id %i, x_neigh =  %f, y_neigh =  %f \n", order, neigh_id+1,  x_neigh, y_neigh);
			//printf(" order = %i neigh_id %i, y_neigh =  %f \n", order, neigh_id+1, y_neigh );


			//учет всех возможных состояний с периодичимкими гран условиями
			if (((x_neigh>r_left)&&(x_neigh<r_right))&&((y_neigh>r_bottom)&&(y_neigh<r_up)))
			{
				ind_pcf = int(round((sqrt((x_center-x_neigh)*(x_center-x_neigh)+(y_center-y_neigh)*(y_center-y_neigh)))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);
				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("1 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf < 0.2/dr) printf("1 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}
			}
			else if  ((x_neigh>r_left)&&(x_neigh<r_right)&&(y_neigh>r_up))
			{
				ind_pcf =  int(round(sqrt((x_center-x_neigh)*(x_center-x_neigh)+(y_center+(y_max-y_neigh))*(y_center+(y_max-y_neigh)))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);

				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("2 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf <  0.2/dr) printf("2 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}

			}
			else if ((x_neigh>r_right)&&(y_neigh>r_up))
			{
				ind_pcf =  int(round(sqrt((x_center+(x_max-x_neigh))*(x_center+(x_max-x_neigh)) +(y_center+(y_max-y_neigh))*(y_center+(y_max-y_neigh)))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);
				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("3 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf <  0.2/dr) printf("3 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}

			}
			else if ((y_neigh>r_bottom)&&(y_neigh<r_up)&&(x_neigh>r_right))
			{
				ind_pcf =  int(round(sqrt((x_max-x_neigh+x_center)*(x_max-x_neigh+x_center)+(y_neigh-y_center)*(y_neigh-y_center))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);
				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("4 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf <  0.2/dr) printf("4 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}
			}
			else if ((x_neigh>r_right)&&(y_neigh<r_bottom))
			{
				ind_pcf =  int(round(sqrt((x_max-x_neigh+x_center)*(x_max-x_neigh+x_center)+(y_neigh+y_max-y_center)*(y_neigh+y_max-y_center))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);
				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("5 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf <  0.2/dr) printf("5 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}
			}
			else if  ((x_neigh>r_left)&&(x_neigh<r_right)&&(y_neigh<r_bottom))
			{
				ind_pcf =  int(round(sqrt((x_center-x_neigh)*(x_center-x_neigh)+(y_max-y_center+y_neigh)*(y_max-y_center+y_neigh))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);
				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("6 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf <  0.2/dr) printf("6 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}
			}
			else if ((x_neigh<r_left)&&(y_neigh<r_bottom))
			{
				ind_pcf =  int(round(sqrt((x_max-x_center+x_neigh)*(x_max-x_center+x_neigh) +(y_max-y_center+y_neigh)*(y_max-y_center+y_neigh))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);
				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("7 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf <  0.2/dr) printf("7 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}
			}
			else if ((y_neigh>r_bottom)&&(y_neigh<r_up)&&(x_neigh<r_left))
			{
				ind_pcf =  int(round(sqrt((x_max-x_center+x_neigh)*(x_max-x_center+x_neigh)+(y_neigh-y_center)*(y_neigh-y_center))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);
				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("8 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf <  0.2/dr) printf("8 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}
			}
			else if ((x_neigh<r_left)&&(y_neigh>r_up))
			{
				ind_pcf =  int(round(sqrt((x_max-x_center+x_neigh)*(x_max-x_center+x_neigh)+(y_center+(y_max-y_neigh))*(y_center+(y_max-y_neigh)))/dr));
				distance = ind_pcf * dr;
				d_pcf[order * len_r + ind_pcf] += 1/(norm_const * distance);
				if (order == 0)
				{
					if (ind_pcf > 5/dr) printf("9 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
					if (ind_pcf <  0.2/dr) printf("9 order =%i,  ind_pcf = %d \n",order, order * len_r + ind_pcf);
				}
			}
		}
	}
}

class MD_State{
	std::fstream 	f;					// fail s vxodnumi dannumi
	std::fstream 	f_vor;					// fail s vxodnumi dannumi

	// ------------------------------
	int 			nparts;					// number of particles in system +
	int				sphere_order;
	double 			r_max;				// maximum distance to pcf
	int 			len_r;				// number of intervals
	int 			num_max_neighbors = 300;

	double 			x_min;				// MD area x_min +
	double 			x_max;				// MD area x_max +
	double 			y_min;				// +
	double 			y_max;				// +
	double 			z_min = -0.2;				// +
	double 			z_max = 0.2;				// +
	double 			*h_xcord;					// massiv dl9 x-coordinat _+
	double 			*h_ycord;					// massiv dl9 y-coordinat +
	double 			*h_pcf_1;					// massiv dl9 pair corrlation fucntion
	double			*h_n_order;					// array for calc number of particles in corr sphere
	int 			*h_neigh_list_all_1;
	//std::vector<std::vector<int>> neigh_list;

	// ------------------------------
	std::string 	s;					// vrenennam0 peremenna9 dl9 chteni9

	// -GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_GPU_------------------------

	dim3    dimGrid_map;        			//
    dim3    dimBlock_map;
	double 			*d_xcord;				// gpu-memory x-coordinata
	double 			*d_ycord;				// gpu-memory y-coordinata
	double 			*d_pcf;					// massiv dl9 pair corrlation fucntion on GPU
	int 			*d_neigh_list_all;					// distance array for pcf on GPU
	int 			*d_neigh_list_all_help;					// distance array for pcf on GPU
	double			*d_n_order;					// array for calc number of particles in corr sphere

public:
	MD_State(std::string &full_name_file, std::string &full_name_file_voronoi, int order, double r_maximum, int len_dist);		// konstructor
	~MD_State();						// destructor
	int Read_Header();					// read Lammps info system
	void Read_State(int flag);					// read new timestep corrd and velocity
	void Read_Voronoi_info(int flag);			// read voronoi neighbors information
	void Calc_PCF();
	void Dump_pcf(std::ofstream& full_name_file);
	void Dump_N_order(std::ofstream& full_name_file, int ts);

};
void MD_State::Calc_PCF(){
	std::cout<<"N_particles = "<<nparts<<"; in Calc_PCF\n";

	cudaMemcpy(d_neigh_list_all, h_neigh_list_all_1, nparts * sphere_order * num_max_neighbors * sizeof(int), cudaMemcpyHostToDevice);
 	cudaMemcpy(d_neigh_list_all_help, h_neigh_list_all_1, nparts * sphere_order * num_max_neighbors * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_xcord, h_xcord, nparts * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ycord, h_ycord, nparts * sizeof(double), cudaMemcpyHostToDevice);


    dimGrid_map.x = static_cast<int>(32);
	dimGrid_map.y = static_cast<int>(25);
	dimGrid_map.z = static_cast<int>(1);

	dimBlock_map.x = static_cast<int>(32);
	dimBlock_map.y = static_cast<int>(1);
	dimBlock_map.z = static_cast<int>(1);

	returnOrderNeighboris<<<dimGrid_map,dimBlock_map>>>(d_neigh_list_all, d_neigh_list_all_help, sphere_order, nparts, num_max_neighbors, d_n_order);//, i);
	cudaDeviceSynchronize();

	cudaMemcpy(h_n_order, d_n_order, sphere_order * nparts * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Calc_PCF 1 : Device Variable:\t%s\n", cudaGetErrorString(cudaGetLastError()));

	//cudaMemcpy(h_neigh_list_all_1, d_neigh_list_all, nparts * sphere_order * num_max_neighbors * sizeof(int), cudaMemcpyDeviceToHost);

	dimGrid_map.x = static_cast<int>(sphere_order);
	dimGrid_map.y = static_cast<int>(1);
	dimGrid_map.z = static_cast<int>(1);

	dimBlock_map.x = static_cast<int>(1);
	dimBlock_map.y = static_cast<int>(1);
	dimBlock_map.z = static_cast<int>(1);
	int iter = 4;
	for(int id_iter = 0; id_iter < iter; id_iter++)
	{
		calc_distance<<<dimGrid_map,dimBlock_map>>>(d_pcf, d_neigh_list_all, d_xcord, d_ycord, x_min, x_max, y_min, y_max, r_max / len_r , len_r, nparts, num_max_neighbors, sphere_order, iter, id_iter);
		cudaDeviceSynchronize();
	}
	printf("Calc_PCF 2 : cudaGetErrorName:\t%s\n", cudaGetErrorName(cudaGetLastError()));



}

MD_State::MD_State(std::string &full_name_file, std::string &full_name_file_voronoi, int order, double r_maximum, int len_dist){		//konstructor

	sphere_order = order;
	r_max =r_maximum;
	len_r =len_dist;
	f.open(full_name_file.c_str());					// open the lammps file trajectory
    nparts=Read_Header();					// the number of particles in system

    h_xcord = new double[nparts];					// vudelenie pam9ti pod massivu
    h_ycord = new double[nparts];					// vudelenie pam9ti pod massivu
    h_n_order = new double[sphere_order * nparts];			//allocate memory for number of neignbors in sphere array
    for(int i = 0; i < sphere_order*nparts; i++)
	{
		h_n_order[i] = 0;
	}

	int id_part;
    for(int i=0;i<nparts;i++){				// chtenie dannux po chasticam
    	f>>id_part;							// chtenie indeksov
		//f>>s;
		f>>h_xcord[id_part - 1];						// chtenie x-koordinatu
		f>>h_ycord[id_part - 1];						// chtenie y-koordinatu
		//f>>s;
		f>>s; //f>>Vx[i];						// chtenie vx-scorostei
        f>>s; //f>>Vy[i];						// chtenie vy-scorostei
        //f>>s;
    }

	double dr = r_max/len_r;
	std::cout<<"dr = "<<dr<<"\n";

	h_pcf_1 = new double[len_r * sphere_order];

	for(int id_r = 0; id_r < sphere_order * len_r; id_r++)
	{
		h_pcf_1[id_r] = 0.0;
	}

	h_neigh_list_all_1 = new int[nparts * sphere_order * num_max_neighbors];

    f_vor.open(full_name_file_voronoi.c_str());

	Read_Voronoi_info(1);

	cudaMalloc((void**)&d_n_order,sphere_order*nparts*sizeof(double));
	cudaMalloc((void**)&d_xcord, nparts*sizeof(double));
	cudaMalloc((void**)&d_ycord, nparts*sizeof(double));
	cudaMalloc((void**)&d_pcf, len_r*sphere_order*sizeof(double));
	cudaMalloc((void**)&d_neigh_list_all, nparts*sphere_order*num_max_neighbors*sizeof(int));
	cudaMalloc((void**)&d_neigh_list_all_help, nparts*sphere_order*num_max_neighbors*sizeof(int));
	cudaDeviceSynchronize();

	cudaMemcpy(d_pcf, h_pcf_1, sphere_order * len_r * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n_order, h_n_order, sphere_order*nparts*sizeof(double), cudaMemcpyHostToDevice);

	printf("MD_State : Device Variable:\t%s\n", cudaGetErrorString(cudaGetLastError()));


}
void MD_State::Read_Voronoi_info(int flag){
	for(int i = 0; i<nparts * sphere_order * num_max_neighbors; i++)
	{
		h_neigh_list_all_1[i] = 0;
	}
 	//memset(h_neigh_list_all_1, 0, nparts * sphere_order * num_max_neighbors);// = new int[nparts * sphere_order * num_max_neighbors];


	int n_rows;
	int id_part;
 	int value;
    f_vor>>s;f_vor>>s; 							// ITEM: TIMESTEP
    f_vor>>s;								// 1000
    f_vor>>s;f_vor>>s;f_vor>>s;f_vor>>s;				// ITEM: NUMBER OF ENTRIES
    f_vor>>n_rows;								// 10000

    //std::vector<std::vector<int>> neigh_list(nparts);
    f_vor>>s;f_vor>>s;f_vor>>s;f_vor>>s;f_vor>>s;f_vor>>s;		// ITEM: BOX BOUNDS pp pp pp
    f_vor>>s;							// chtenie razmerov x_min
    f_vor>>s;							// chtenie razmerov x_mox
    f_vor>>s;							// chtenie razmerov
    f_vor>>s;							// chtenie razmerov
    f_vor>>s;//z_min;								// propusk z-coordinat
    f_vor>>s;//z_max;								// propusk z-coordinat
    f_vor>>s;f_vor>>s;f_vor>>s;f_vor>>s;//f_vor>>s;f_vor>>s; //f>>s;f>>s;	// chtenie
  //ITEM:     ENTRIES c_6[1]   c_6[2]

	int counters[nparts] = {0};
	for(int i=0;i < n_rows;i++)
	{				// chtenie dannux po chasticam
    	//f_vor>>s;							// chtenie indeksov
		f_vor>>id_part;						//индекс центральной частиц c_6[1]
		f_vor>>value;						//индекс соседа по ячейке вороного c_6[2]
		if (value > 0)
		{
			//neigh_list[id_part - 1].push_back(value);						// chtenie x-koordinatu
			h_neigh_list_all_1[num_max_neighbors * (id_part - 1) * sphere_order + counters[id_part - 1]] = value;
			counters[id_part - 1]++;
		}//f_vor>>s;  											// c_6[3]
    }
	if (flag == 0)
	{
		for(int id_part = 0; id_part < nparts; id_part++)
		{

			h_n_order[sphere_order*id_part + 0] += double(counters[id_part]) / nparts;
		}
		std::cout<<"in flag:   "<<h_n_order[0]<<";  "<<h_n_order[sphere_order*2]<<"\n";

	}



}
int MD_State::Read_Header(){
    int     	n;
    f>>s;f>>s; 							// ITEM: TIMESTEP
    f>>s;								// 0
    f>>s;f>>s;f>>s;f>>s;				// ITEM:NUMBER OF ATOMS
    //std::cout<<s<<"\n";
    f>>n;								// 10000
    //std::cout<<"n="<<n<<"\n";
    f>>s;f>>s;f>>s;f>>s;f>>s;f>>s;		// ITEM: BOX BOUNDS pp pp pp
    f>>x_min;							// chtenie razmerov
    f>>x_max;							// chtenie razmerov
    f>>y_min;							// chtenie razmerov
    f>>y_max;							// chtenie razmerov
    f>>s;//z_min;								// propusk z-coordinat
    f>>s;//z_max;								// propusk z-coordinat
    f>>s;f>>s;f>>s;f>>s;f>>s;f>>s;f>>s;//f>>s;f>>s;	// chtenie musora
  //ITEM ATOMS id    x    y   vx  vy
    return n;
}

void MD_State::Read_State(int flag){

    Read_Header();
	Read_Voronoi_info(flag);
    int id_part;
    for(int i=0;i<nparts;i++){
    	f>>id_part;

    	if(id_part<1){
    		std::cout<<id_part<<"\n";
    	}
		//read data coordinat
    	f>>h_xcord[id_part - 1];
    	f>>h_ycord[id_part - 1];

		// read velocity data
		f>>s; //h_vx[id_part - 1];
		f>>s; //h_vy[id_part - 1];

    }
}


void MD_State::Dump_pcf(std::ofstream& full_name_file){
	cudaMemcpy(h_pcf_1, d_pcf, sphere_order * len_r * sizeof(double), cudaMemcpyDeviceToHost);
 	printf("Dump_pcf: cudaGetErrorName:\t%s\n", cudaGetErrorString(cudaGetLastError()));
	for(int id_r =0 ; id_r < len_r; id_r++)
	{
		for(int order = 0; order < sphere_order; order++)
		{
			full_name_file<<h_pcf_1[id_r + order*len_r]<<" ";
		}
		full_name_file<<"\n";
	}

	for(int id_r = 0; id_r < sphere_order * len_r; id_r++)
	{
		h_pcf_1[id_r] = 0.0;
	}

}
void MD_State::Dump_N_order(std::ofstream& full_name_file, int ts){
	//cudaMemcpy(h_n_order, d_n_order, sphere_order * nparts * sizeof(double), cudaMemcpyDeviceToHost);
 	printf("Dump_N_order: cudaGetErrorName:\t%s\n", cudaGetErrorString(cudaGetLastError()));
	double array[sphere_order] = {0};
	for(int order = 0; order < sphere_order; order++)
	{
		for(int i = 0; i < nparts; i ++)
		{
			array[order] += h_n_order[sphere_order*i + order]/ts;
		}
	}
	for(int order = 0; order < sphere_order; order++)
	{
		full_name_file<<array[order] / (order+1)<<"\n";
	}

	for(int id_r = 0; id_r < sphere_order * nparts; id_r++)
	{
		h_n_order[id_r] = 0.0;
	}

}

MD_State::~MD_State(){
	f.close();								// zakrutie vxodnogo faila
	f_vor.close();							// zakrutie vxodnogo faila
	delete [] h_xcord;						// osvoboshdenie pam9ti
	delete [] h_ycord;						// osvoboshdenie pam9ti
	delete [] h_pcf_1;
	delete [] h_neigh_list_all_1;

	cudaFree(d_xcord);
    cudaFree(d_ycord);
    cudaFree(d_neigh_list_all);
    cudaFree(d_neigh_list_all_help);
	cudaFree(d_pcf);

	//cudaDeviceReset();
}
int main(){
	std::string full_path, full_path2,full_save,full_save2;
    std::string potentials[] = {"LJ", "IPL3", "IPL9", "IPL12", "IPL18"};
    std::string path = "/run/media/artur/Новый том/Liquid_data_2020_2021/MD_data/lammps_trj_";
	std::string save_path = "/run/media/artur/Новый том/Liquid_data_2020_2021/MD_data/result_new/txt_data/";
//	cudaDeviceReset();
	int ts_num = 0;

	for(int pot=0; pot<5; pot++)
	{
		for(int temp = 0; temp<6; temp++)
		{
			ts_num = 0;
			high_resolution_clock::time_point start;
			high_resolution_clock::time_point now;
			start=high_resolution_clock::now();
			double tnow;
			full_path = path + potentials[pot] + "/" + "d_" + potentials[pot] + "_T_" + std::to_string(temp+1) + "_rho_1_ts.lammpstrj";
			full_path2 =path + potentials[pot] + "/" + "d_" + potentials[pot] + "_T_" + std::to_string(temp+1) + "_rho_1_ts.neighbors";
 			//full_path = "/run/media/artur/Новый том/Liquid_data_2020_2021/MD_data/d_LJ_T_1_rho_1_ts.lammpstrj";
 			//full_path2 ="/run/media/artur/Новый том/Liquid_data_2020_2021/MD_data/d_LJ_T_1_rho_1_ts.neighbors";
			std::cout<<full_path<<"\n";
			std::cout<<full_path2<<"\n";

			full_save = save_path + "FINUL_pcf_" + potentials[pot] + "_T_" + std::to_string(temp+1) + "CUDA.txt";
			full_save2 = save_path + "FINUL_N(order)_" + potentials[pot] + "_T_" + std::to_string(temp+1) + "CUDA.txt";
			std::cout<<full_save<<"\n";
			std::ofstream fout_L(full_save);      // fail dl9 vuvoda dannux v lammps formate
			std::ofstream fout_N(full_save2);      // fail dl9 vuvoda dannux v lammps formate

			MD_State MD(full_path, full_path2, 20, 35, 3000);

			for(int ii=1;ii<2900;ii++)
			{												// skip first frames
				MD.Read_State(ii%40);											// chtenie shaga
				//std::cout<<ii<<"\n";
				if (ii%40 == 0)
				{
					std::cout<<"ii = "<<ii<<"; "<<potentials[pot]<<"; T = "<<std::to_string(temp+1)<<"\n";
					MD.Calc_PCF();
					ts_num++;
				}
			}
			std::cout<<"\n";
			MD.Dump_pcf(fout_L);
			MD.Dump_N_order(fout_N,ts_num);
			now = high_resolution_clock::now();
			tnow=std::chrono::duration<double>(now-start).count();
			std::cout <<"time = "<< tnow << std::endl;




		}

	}

    return(0);
}




/* ================MYSOR===============
 * int nn;
	for(int id_part = 0; id_part < 1; id_part++)
	{
		for(int order = 0; order < sphere_order; order ++)
		{
			int count = 0;
			int id_CENTER_part_in_array = id_part * num_max_neighbors * sphere_order;			//индекс центральной частицы в массиве данных
			for(int neigh = 0; h_neigh_list_all_1[id_CENTER_part_in_array + neigh + num_max_neighbors * (order )]; neigh++)
			{ //neigh - порядок соседней частицы на пред сфере (1-я частица 2-я частьиц итл)

				int neigh_id = h_neigh_list_all_1[id_CENTER_part_in_array + neigh + num_max_neighbors * order];
				std::cout<<neigh_id<<" ";
				nn = neigh;
			}
			std::cout<<"order = "<<order+1<<"; neigh = "<<nn<<"; n/order = "<<double(nn)/(order+1)<<"\n";
		}

	}


 */
