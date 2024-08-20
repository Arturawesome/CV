/*
 ============================================================================
Name       : flow_anal.cu
Author     : Nasyrov A.D.

Copyright  : Your copyright notice
Description: analysis of active sysytem with periodic flow force.
            In this C++/CUDA program we calc
                                    1) impulse moment of rotating particles
                                    2) velocity-celocity correlation for estimating of mictoflows
                                    3) diffusion of system: gistogram of MSD and sigma^2(t) dependencies

Technology : CUDA, Object Oriented Programming

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

extern "C" __global__ void VelVelCorrelation(double *d_vv_cor, double *d_xcord_now, double *d_ycord_now, double *d_vx_now, double *d_vy_now, int nparts, double x_box, double r_cut, double dr_in_vv, int n_elem_in_vv, double *d_vv_angles, double *d_count_for_vv_angles)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс уцентральной частицы (-1)

	if (id_part < nparts)
    {
        double vx0 = 1, vy0 = 0;
        double x_center = d_xcord_now[id_part], y_center = d_ycord_now[id_part];
        double r_right, r_left, r, drx, dry, x_rot, y_rot;
        int id_vv_x, id_vv_y;

        double vx_center = d_vx_now[id_part], vy_center = d_vy_now[id_part];
        double dot_product, vx_neigh, vy_neigh;
        double rot_angle, vv_corr_ij;
        double vx_center_rot, vy_center_rot, vx_neigh_rot, vy_neigh_rot;

        r_right = x_center+x_box/2.1;
        r_left = x_center-x_box/2.1;

        for(int neigh = 0; neigh < nparts; neigh++)
        {
            if(neigh == id_part) continue;
            double x_neigh = d_xcord_now[neigh];
            double y_neigh = d_ycord_now[neigh];
            dry = y_neigh - y_center;

            if(x_neigh > r_left && x_neigh < r_right) drx = x_neigh - x_center;
            else if(x_neigh < r_left) drx = x_neigh + x_box - x_center;
            else drx = x_neigh - (x_box + x_center);

            r = sqrt(drx * drx + dry * dry);

            if(r > r_cut)
            {
                continue;
            }
            else
            {
                dot_product = (vx0*vx_center+vy0*vy_center)/(sqrt(vx0*vx0+vy0*vy0)*sqrt(vx_center*vx_center+vy_center*vy_center));

                if (dot_product<-1)
                {
                    dot_product=-1;
                }
                else if(dot_product> 1)
                {
                    dot_product= 1;
                }
                // определение угля от 0 до 2pi
                // acos возвращает значение отнуля до пи.
                rot_angle = acos(dot_product);
                if(rot_angle > M_PI / 2)
                {
                    if (vy_center < 0)
                    {
                        rot_angle = abs(rot_angle - (M_PI + M_PI /2 ));
                    }
                    else
                    {
                        rot_angle -= M_PI/2 ;
                    }
                }
                else if (rot_angle < M_PI / 2)
                {
                    if (vy_center < 0)
                    {
                        rot_angle = abs(rot_angle - (M_PI + M_PI /2 ));
                    }
                    else
                    {
                        rot_angle = 2*M_PI - (M_PI/2 - rot_angle);
                    }
                }
                vx_neigh = d_vx_now[neigh];
                vy_neigh = d_vy_now[neigh];

                vx_center_rot = vx_center * cos(rot_angle) + vy_center * sin(rot_angle);
                vy_center_rot = vy_center * cos(rot_angle) - vx_center * sin(rot_angle);

                vx_neigh_rot = vx_neigh * cos(rot_angle) + vy_neigh * sin(rot_angle);
                vy_neigh_rot = vy_neigh * cos(rot_angle) - vx_neigh * sin(rot_angle);

                vv_corr_ij = vx_center_rot * vx_neigh_rot + vy_center_rot * vy_neigh_rot;

                x_rot = drx * cos(rot_angle) + dry * sin(rot_angle);
                y_rot = dry * cos(rot_angle) - drx * sin(rot_angle);

                id_vv_x = int(floor(x_rot / dr_in_vv)) + n_elem_in_vv;
                id_vv_y = int(floor(y_rot / dr_in_vv)) + n_elem_in_vv;

                d_vv_cor[2 * n_elem_in_vv * id_vv_y +  id_vv_x ] += vv_corr_ij;

                // part for calc average angle betwen velocity vectors
                dot_product = (vx_neigh*vx_center+vy_neigh*vy_center)/(sqrt(vx_neigh*vx_neigh+vy_neigh*vy_neigh)*sqrt(vx_center*vx_center+vy_center*vy_center));

                if (dot_product<-1)
                {
                    dot_product=-1;
                }
                else if(dot_product> 1)
                {
                    dot_product= 1;
                }
                rot_angle = acos(dot_product);
                id_vv_x = int(floor(r / dr_in_vv));

                if(id_part * n_elem_in_vv + id_vv_x > nparts * n_elem_in_vv - 1)
                {
                     printf("yes border ");
                }

                d_vv_angles[id_part * n_elem_in_vv + id_vv_x] += rot_angle;
                d_count_for_vv_angles[id_part * n_elem_in_vv + id_vv_x] += 1;
            }
        }
    }
}

extern "C" __global__ void CalcImpulsMomentGPU(double *d_xcord_now, double *d_ycord_now, double *d_vx_now, double *d_vy_now, double *d_imp_of_part, int nparts, double x_box)
{
    // calac made relative each particle
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс уцентральной частицы (-1)
	double drx_box = x_box / 4;
    double xpart;

	if (id_part < nparts)
    {
//         if ( d_xcord_now[id_part] > 0)
//         {
//             d_imp_of_part[id_part] += (d_xcord_now[id_part] * d_vy_now[id_part] - d_ycord_now[id_part] * d_vx_now[id_part]);
//         }
//         else
//         {
//             d_imp_of_part[id_part] += 0;
//         }
        if(d_xcord_now[id_part] > 0 && d_xcord_now[id_part] < drx_box)
        {
            xpart = d_xcord_now[id_part] - drx_box/2;
            d_imp_of_part[id_part] += (xpart * d_vy_now[id_part] - d_ycord_now[id_part] * d_vx_now[id_part]);
        }
        else if(d_xcord_now[id_part] < 0 && d_xcord_now[id_part] > -drx_box)
        {
            xpart = d_xcord_now[id_part] + drx_box/2;
            d_imp_of_part[id_part] -= (xpart * d_vy_now[id_part] - d_ycord_now[id_part] * d_vx_now[id_part]);
        }
        else
        {
            d_imp_of_part[id_part] += 0;
        }
    }
}

extern "C" __global__ void ReturnSigma2tDependences(double *d_sigma2t, double *d_sigma2t_itog, int len_ts_for_sigm2t, int nparts)
{
    //calc relative of time
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	int id_time = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс времени
    if (id_time < len_ts_for_sigm2t)
    {
        for(int id_part = 0; id_part < nparts; id_part++)
        {
            d_sigma2t_itog[id_time] += (d_sigma2t[id_part * len_ts_for_sigm2t + id_time] / nparts);
        }
        //printf("( %f )", d_sigma2t_itog[id_time]);
    }

}

extern "C" __global__ void CalcMsd(double *d_xcord_now, double *d_ycord_now, double *d_xcord_past, double *d_ycord_past,  double *d_sigma2tx, double *d_sigma2ty, double *d_sigma2t, double x_box, double *d_sigma_gist, int nparts, double dr_in_gist, int len_ts_for_sigm2t, int ts, double *d_flow_estimate_per_particle)
{
    // calc relative of particles
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс уцентральной частицы (-1)

	if (id_part < nparts)
    {
        double drx, dry, r_right, r_left, x_now, x_past, y_now, y_past;
        x_now = d_xcord_now[id_part];
        y_now = d_ycord_now[id_part];
        x_past = d_xcord_past[id_part];
        y_past = d_ycord_past[id_part];
        r_right = x_now+x_box/2.1;
        r_left = x_now-x_box/2.1;
        dry = y_now - y_past;

        if(x_past > r_left && x_past < r_right)
        {
            drx = x_now - x_past;
        }
        else if(x_past < r_left)
        {
            drx = x_box - x_now + x_past;
        }
        else
        {
            drx = x_box - x_past + x_now;
        }

        //== displacment each particles in present timepstep =======================
        d_sigma2tx[id_part * len_ts_for_sigm2t + ts] = drx;
        d_sigma2ty[id_part * len_ts_for_sigm2t + ts] = dry;

        //== id displametnt in gistogram =======================
        d_sigma_gist[id_part * len_ts_for_sigm2t + ts] = int(floor(sqrt(drx * drx + dry * dry) / dr_in_gist));

        if(ts % len_ts_for_sigm2t == (len_ts_for_sigm2t - 1))
        {
            drx = d_sigma2tx[id_part * len_ts_for_sigm2t + 0];
            dry = d_sigma2ty[id_part * len_ts_for_sigm2t + 0];
            d_sigma2t[id_part * len_ts_for_sigm2t + 0] = sqrt(drx * drx + dry * dry);

            for(int id_ts = 1; id_ts < len_ts_for_sigm2t; id_ts++)
            {
                drx += d_sigma2tx[id_part * len_ts_for_sigm2t + id_ts];
                dry += d_sigma2ty[id_part * len_ts_for_sigm2t + id_ts];
                d_sigma2t[id_part * len_ts_for_sigm2t + id_ts] = sqrt(drx * drx + dry * dry);
            }
        }
        // ============ flow calculate ===================
        r_left = -0.5 * x_box;
        r_right = 0.5 * x_box;
        drx =  0.5 * x_box / 4.5;
        /* по середине и на границах моделирований поток частиц совпадает по направлению п
         * на середеинных участках поток противоположный, поэтому знаки отличаются
         *
         * площадка прохода расположена на координате y=0,
         * порэтому при пересечении ее произведение координат будет разнознаковым
         */
        if ((x_now > 0 - drx) && (x_now < 0 + drx) && (x_past > 0 - drx) && (x_past < 0 + drx))
        {

            if (y_now * y_past < 0)
            {
                if (y_now > 0)
                {
                    d_flow_estimate_per_particle[id_part] -= 1;
                }
                else
                {
                    d_flow_estimate_per_particle[id_part] += 1;
                }
            }
        }
        if ((x_now < r_left + drx) && (x_now > r_right - drx) && (x_past < r_left + drx) && (x_past > r_right - drx) )
        {
            if (y_now * y_past < 0)
            {
                if (y_now > 0)
                {
                    d_flow_estimate_per_particle[id_part] -= 1;
                }
                else
                {
                    d_flow_estimate_per_particle[id_part] += 1;
                }
            }
        }

        if ((x_now > r_right/2  - drx) && (x_now < r_right/2 + drx) && (x_past > r_right/2  - drx) && (x_past < r_right/2 + drx))
        {
            if (y_now * y_past < 0)
            {
                if (y_now > 0)
                {
                    d_flow_estimate_per_particle[id_part] += 1;
                }
                else
                {
                    d_flow_estimate_per_particle[id_part] -= 1;
                }
            }
        }
        if ((x_now > r_left/2 - drx) && (x_now < r_left/2 + drx) && (x_past > r_left/2 - drx) && (x_past < r_left/2 + drx))
        {
            if (y_now * y_past < 0)
            {
                if (y_now > 0)
                {
                    d_flow_estimate_per_particle[id_part] += 1;
                }
                else
                {
                    d_flow_estimate_per_particle[id_part] -= 1;
                }
            }
        }
    }
}
























class Flow{
    std::fstream 	f_flow;					// filе with flow grid data
	std::fstream 	f_trj;					// filе with lammps f_trj
	std::string 	s;					// vrenennam0 peremenna9 dl9 chteni9
    int     nparts;	            			// number of particles in system (lammps trj file)
    int     x_grid;
    int     y_grid;
    int     n_el_gist;
    int     len_ts_for_sigm2t;
    int     n_elem_in_vv;
    double  r_cut;
    double  flow_estimate;
    double  *h_flow_estimate_per_particle;                   // переменная для оценки потока частиц через площадку в центре моделировани


    double  *h_vx_grid;                          // vx grid on host
    double  *h_vy_grid;                          // vy grid on host
    double  *h_vabs_grid;                        // abs v grid on host
    double  *h_lx_grid;                          // arrray for impuls moment x
    double  *h_ly_grid;                          // arrray for impuls moment y


    double  *h_sigma_gist2;
    double  *h_sigma_gist;                      // mean square displacment of praticle for calc diffusion (GISTOGRAM OF displacment)
    double  *h_xcord;
    double  *h_ycord;
    double  *h_vx;
    double  *h_vy;
    double  *h_imp_of_part;                     // impuls moment of each paticles
    double  *h_sigma2t_itog;

    double  *h_vv_cor;
    double  dr_in_vv;
    double  *h_vv_angles;
    double  *h_count_for_vv_angles;
    double  *h_disp_of_angles;
    double  *h_vv_angles_r;
    double  *h_vv_disp_angles_r;



    double  dr_in_gist;                              // step in r for diffusion gistogram
    std::vector<std::vector<double>> sigma2tx;
    std::vector<std::vector<double>> sigma2ty;
    std::vector<std::vector<double>> sigma2t;

    double x_min, x_max, y_min, y_max;
    double  act;


    // ======GPU======GPU======GPU======GPU======GPU======GPU======GPU======GPU======GPU
    dim3    dimGrid_map;        			//
    dim3    dimBlock_map;

    double  *d_vx_now;
    double  *d_vy_now;
    double  *d_vx_grid;                          // vx grid on device
    double  *d_vy_grid;                          // vy grid on device
    double  *d_vabs_grid;                        // abs v grid on device
    double  *d_lx_grid;                          // arrray for impuls moment x
    double  *d_ly_grid;                          // arrray for impuls moment y

    double  *d_imp_of_part;                     // impuls moment of each paticles
    double  *d_vv_cor;
    double  *d_vv_angles;
    double  *d_count_for_vv_angles;

    double  *d_xcord_now;
    double  *d_ycord_now;
    double  *d_xcord_past;
    double  *d_ycord_past;

    double  *d_sigma_gist;                       // mean square displacment of praticle for calc diffusion gistogram
    double  *d_sigma2tx;                        // смещение чатиц на таймтепах размер nparts * len_ts_for_sigm2t
    double  *d_sigma2ty;
    double  *d_sigma2t;
    double  *d_sigma2t_itog;                    // итог по вычислению сигма, размер len_ts_for_sigm2t

    double  *d_flow_estimate_per_particle;                   // переменная для оценки потока частиц через площадку в центре

public:
    Flow(std::string &full_name_file, std::string &full_name_file_flow, double r_c_dif_gist, int n_in_gist, int len_ts_sigma2t,double r_cut_vv, int n_el_vv);
    ~Flow();
    int Read_Header();
    void Read_State();
    void Read_flow_info();
    void Read_flow_header();

    void CalcMsdGist(int ts);
    void ReturnGist(int ts);
    void CalcImpulsMomet();
    void CalcAnglesDispersion();

    void print_array_2D(int num_str, int num_col, double *array);
    void print_array_1D(int num_str,double *array);
    void WriteGist(std::ofstream& full_name_file, int counter);
    void WriteSigma2t(std::ofstream& full_name_file, int counter);
    void WriteImpulseMoment(std::ofstream& full_name_file, int counter);
    void WriteVelcityCorrelation(std::ofstream& full_name_file, int counter);
    void WriteFlowEstimate(std::ofstream& full_name_file, int counter);
    void WriteAnglesInfo(std::ofstream& full_name_file1, std::ofstream& full_name_file2, int counter);

};

void Flow::CalcImpulsMomet()
{
    // func too calc impuls moment in system
   // double x0 = 0, y0 = 0, z0 = 0;      // point relative to which the calculation is made
    cudaMemcpy(d_vx_now, h_vx, nparts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy_now, h_vy, nparts * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    CalcImpulsMomentGPU<<<dimGrid_map, dimBlock_map>>>(d_xcord_now, d_ycord_now, d_vx_now, d_vy_now, d_imp_of_part, nparts,  (x_max - x_min));
    cudaDeviceSynchronize();

    VelVelCorrelation<<<dimGrid_map, dimBlock_map>>>(d_vv_cor, d_xcord_now, d_ycord_now, d_vx_now, d_vy_now, nparts, (x_max - x_min), r_cut, dr_in_vv, n_elem_in_vv, d_vv_angles, d_count_for_vv_angles);
    cudaDeviceSynchronize();


}
void Flow::CalcAnglesDispersion()
{
    double elem;
    std::cout<<"CalcAnglesDispersion\n";

    cudaMemcpy(h_vv_angles, d_vv_angles, n_elem_in_vv * nparts * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_count_for_vv_angles, d_count_for_vv_angles, n_elem_in_vv * nparts * sizeof(double), cudaMemcpyDeviceToHost);
    printf("CalcAnglesDispersion : cudaGetErrorName:\t%s\n", cudaGetErrorName(cudaGetLastError()));

    for(int id_r = 0; id_r < n_elem_in_vv; id_r++)
    {
        for (int id_part = 0; id_part < nparts; id_part++ )
        {
            if (h_count_for_vv_angles[id_part * n_elem_in_vv + id_r] > 0)
            {
                h_vv_angles_r[id_r] += (h_vv_angles[id_part * n_elem_in_vv + id_r] / h_count_for_vv_angles[id_part * n_elem_in_vv + id_r]);
                //std::cout<< id_part * n_elem_in_vv + id_r<<"\n";
            }
        }

        h_vv_angles_r[id_r] /= nparts;

        for (int id_part = 0; id_part < nparts; id_part ++)
        {
            if (h_count_for_vv_angles[id_part * n_elem_in_vv + id_r] > 0)
            {
                elem = h_vv_angles[id_part * n_elem_in_vv + id_r] / h_count_for_vv_angles[id_part * n_elem_in_vv + id_r];
                h_vv_disp_angles_r[id_r] += ((elem -  h_vv_angles_r[id_r]) * (elem -  h_vv_angles_r[id_r]) / nparts);
            }

        }
    }
}



void Flow::CalcMsdGist(int ts)
{
    // func to calc msd gist
    cudaMemcpy(d_xcord_now, h_xcord, nparts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ycord_now, h_ycord, nparts * sizeof(double), cudaMemcpyHostToDevice);

    if (ts > 0)
    {
        CalcMsd<<<dimGrid_map, dimBlock_map>>>(d_xcord_now, d_ycord_now, d_xcord_past, d_ycord_past, d_sigma2tx, d_sigma2ty, d_sigma2t, (x_max - x_min), d_sigma_gist, nparts, dr_in_gist, len_ts_for_sigm2t, ts-1, d_flow_estimate_per_particle);

        cudaDeviceSynchronize();
        if(ts % len_ts_for_sigm2t == 0)
        {
            cudaMemcpy(h_sigma_gist2, d_sigma_gist, nparts * len_ts_for_sigm2t * sizeof(double), cudaMemcpyDeviceToHost);

            ReturnSigma2tDependences<<<dimGrid_map, dimBlock_map >>>(d_sigma2t, d_sigma2t_itog, len_ts_for_sigm2t, nparts);

            cudaDeviceSynchronize();
            ReturnGist(ts);
        }
    }
    cudaMemcpy(d_xcord_past, h_xcord, nparts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ycord_past, h_ycord, nparts * sizeof(double), cudaMemcpyHostToDevice);
}

void Flow::ReturnGist(int ts)
{
    int id_gist;
    for(int id_part = 0; id_part < nparts; id_part++)
    {
        for(int id_ts = 0; id_ts < len_ts_for_sigm2t; id_ts ++)
        {
            id_gist = h_sigma_gist2[id_part * len_ts_for_sigm2t + id_ts];
            h_sigma_gist[id_gist] += 1;
        }
    }
}

Flow::Flow(std::string &full_name_file, std::string &full_name_file_flow, double r_c_dif_gist, int n_in_gist, int len_ts_sigma2t, double r_cut_vv, int n_el_vv){		//konstructor
    n_el_gist = n_in_gist;
    r_cut = r_cut_vv;
    n_elem_in_vv = n_el_vv;
    dr_in_vv = r_cut / n_elem_in_vv;

    h_vv_cor = new double[4 * n_elem_in_vv * n_elem_in_vv];
    for(int i = 0; i < 4 * n_elem_in_vv * n_elem_in_vv; i ++)
    {
        h_vv_cor[i] = 0;
    }

    h_vv_angles_r = new double[n_elem_in_vv];
    h_vv_disp_angles_r = new double[n_elem_in_vv];
    for(int i = 0; i < n_elem_in_vv; i++)
    {
        h_vv_angles_r[i] = 0;
        h_vv_disp_angles_r[i] = 0;
    }

    len_ts_for_sigm2t = len_ts_sigma2t -1;
	f_trj.open(full_name_file.c_str());					// open the lammps file trajectory
    nparts = Read_Header();					        // number of particles in system

    h_flow_estimate_per_particle =  new double[nparts];
    for(int i = 0; i < nparts; i++)
    {
        h_flow_estimate_per_particle[i] = 0;
    }

    h_vv_angles = new double[n_elem_in_vv * nparts];
    h_count_for_vv_angles = new double[n_elem_in_vv * nparts];

    for(int i = 0; i < n_elem_in_vv * nparts; i++)
    {
        h_vv_angles[i] = 0.0;
        h_count_for_vv_angles[i] = 0.0;
    }

    h_xcord = new double[nparts];					// allocate memory on host for coord
    h_ycord = new double[nparts];					// allocate memory on host for coord

    h_vx = new double[nparts];					// allocate memory on host for velocity
    h_vy = new double[nparts];					// allocate memory on host for velocity
    h_imp_of_part = new double[nparts];

    for(int i = 0; i < nparts; i ++)
    {
        h_imp_of_part[i] = 0;
    }

    h_sigma2t_itog = new double[len_ts_for_sigm2t];
    h_sigma_gist2 = new double[nparts * len_ts_for_sigm2t];
    h_sigma_gist = new double[n_in_gist];
    for(int i= 0; i < n_in_gist; i++)
    {
        h_sigma_gist[i] = 0;
    }
    for(int i= 0; i < len_ts_for_sigm2t; i++)
    {
        h_sigma2t_itog[i] = 0;
    }

    dr_in_gist = r_c_dif_gist / n_in_gist;

//     std::vector<std::vector<double>> sigma2tx(nparts);
//     std::vector<std::vector<double>> sigma2ty(nparts);
//     std::vector<std::vector<double>> sigma2t(len_ts_for_sigm2t);

    std::cout<<"nparts = "<<nparts<<"\n";
	int id_part;
    for(int i = 0; i<nparts; i++)
    {				     // read psticle data
        f_trj>>id_part;							     // read id particle
        f_trj>>h_xcord[id_part];						// read x-koordinatu
        f_trj>>h_ycord[id_part];						// read y-koordinatu
        f_trj>>h_vx[id_part];						// read vx-velocity
        f_trj>>h_vy[id_part];						// read vy-velocity
    }

    // allocate memort on host
    f_flow.open(full_name_file_flow.c_str());
    Read_flow_header();
    std::cout<<"x_grid, y_grid = "<<x_grid<<" "<<y_grid<<"\n";

    h_vx_grid = new double[x_grid * y_grid];
    h_vy_grid = new double[x_grid * y_grid];

    h_lx_grid = new double[x_grid * y_grid];
    h_ly_grid = new double[x_grid * y_grid];

    Read_flow_info();

    // allocate memort on device

    cudaMalloc((void**)&d_flow_estimate_per_particle, nparts * sizeof(double));

    cudaMalloc((void**)&d_vx_grid, x_grid * y_grid*sizeof(double));
    cudaMalloc((void**)&d_vy_grid, x_grid * y_grid*sizeof(double));

    cudaMalloc((void**)&d_lx_grid, x_grid * y_grid*sizeof(double));
    cudaMalloc((void**)&d_ly_grid, x_grid * y_grid*sizeof(double));

    cudaMalloc((void**)&d_xcord_now, nparts * sizeof(double));
    cudaMalloc((void**)&d_ycord_now, nparts * sizeof(double));

    cudaMalloc((void**)&d_xcord_past, nparts * sizeof(double));
    cudaMalloc((void**)&d_ycord_past, nparts * sizeof(double));

    cudaMalloc((void**)&d_vx_now, nparts * sizeof(double));
    cudaMalloc((void**)&d_vy_now, nparts * sizeof(double));

    cudaMalloc((void**)&d_imp_of_part, nparts * sizeof(double));
    cudaMalloc((void**)&d_vv_cor, 4 * n_elem_in_vv * n_elem_in_vv * sizeof(double));

    cudaMalloc((void**)&d_vv_angles, n_elem_in_vv * nparts * sizeof(double));
    cudaMalloc((void**)&d_count_for_vv_angles, n_elem_in_vv * nparts * sizeof(double));

    cudaMalloc((void**)&d_sigma2tx, nparts * len_ts_for_sigm2t * sizeof(double));
    cudaMalloc((void**)&d_sigma2ty, nparts * len_ts_for_sigm2t * sizeof(double));
    cudaMalloc((void**)&d_sigma2t, nparts * len_ts_for_sigm2t * sizeof(double));
    cudaMalloc((void**)&d_sigma2t_itog, len_ts_for_sigm2t * sizeof(double));
    cudaMalloc((void**)&d_sigma_gist, nparts * len_ts_for_sigm2t * sizeof(double));

    cudaMemcpy(d_flow_estimate_per_particle, h_flow_estimate_per_particle,nparts * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sigma2t_itog, h_sigma2t_itog, len_ts_for_sigm2t * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imp_of_part, h_imp_of_part, nparts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vv_cor, h_vv_cor, 4 * n_elem_in_vv * n_elem_in_vv * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_count_for_vv_angles, h_count_for_vv_angles, n_elem_in_vv * nparts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vv_angles, h_vv_angles, n_elem_in_vv * nparts * sizeof(double), cudaMemcpyHostToDevice);

    dimBlock_map.x = static_cast<int>(32);
    dimBlock_map.y = static_cast<int>(1);
    dimBlock_map.z = static_cast<int>(1);

    dimGrid_map.x = static_cast<int>(32);
    dimGrid_map.y = static_cast<int>(1);
    dimGrid_map.z = static_cast<int>(1);


    printf("Flow : cudaGetErrorName:\t%s\n", cudaGetErrorName(cudaGetLastError()));
}
Flow::~Flow()
{
   f_trj.close();
    f_flow.close();
    delete [] h_vx_grid;
    delete [] h_vy_grid;

    delete [] h_lx_grid;
    delete [] h_ly_grid;

    delete [] h_sigma_gist;
    delete [] h_xcord;
    delete [] h_ycord;
    delete [] h_vx;
    delete [] h_vy;

    cudaFree(d_vx_grid);
    cudaFree(d_vy_grid);
    //cudaFree(d_vabs_grid);
    cudaFree(d_lx_grid);
    cudaFree(d_ly_grid);

    cudaFree(d_sigma2tx);
    cudaFree(d_sigma2ty);
    cudaFree(d_sigma2t);
    cudaFree(d_sigma_gist);
    cudaFree(d_sigma2t_itog);

    cudaFree(d_xcord_now);
    cudaFree(d_ycord_now);
    cudaFree(d_xcord_past);
    cudaFree(d_ycord_past);

}
int Flow::Read_Header(){
    int     	n;
    f_trj>>s;f_trj>>s; 							// ITEM: TIMESTEP
    f_trj>>s;								// 0
    f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;				// ITEM:NUMBER OF ATOMS
    f_trj>>n;								// 10000
    f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;		// ITEM: BOX BOUNDS pp pp pp
    f_trj>>x_min;							// chtenie razmerov
    f_trj>>x_max;							// chtenie razmerov
    f_trj>>y_min;							// chtenie razmerov
    f_trj>>y_max;							// chtenie razmerov
    f_trj>>s;//z_min;								// propusk z-coordinat
    f_trj>>s;//z_max;								// propusk z-coordinat
    f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;//f_trj>>s;f_trj>>s;	// chtenie musora
    //ITEM ATOMS id    x    y   vx  vy
    return n;
}

void Flow::Read_State(){
    Read_flow_header();
    Read_flow_info();
    Read_Header();
    int id_part;
    for(int i = 0; i < nparts; i++){
    	f_trj>>id_part;

        //read data coordinat
    	f_trj>>h_xcord[id_part];
    	f_trj>>h_ycord[id_part];

        // read velocity data
		f_trj>>h_vx[id_part];
		f_trj>>h_vy[id_part];
    }
}
void Flow::Read_flow_header()
{

    f_flow>>s;f_flow>>s; 							//ITEM: TIMESTEP
    f_flow>>s;								// 1000
    f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;				// ITEM: NUMBER OF ATOMS
    f_flow>>s;								// 924

    //std::vector<std::vector<int>> neigh_list(nparts);
    f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;		//ITEM: BOX BOUNDS pp pp grid_size_xy
    f_flow>>s;							// chtenie razmerov x_min
    f_flow>>s;							// chtenie razmerov x_mox
    f_flow>>s;							// chtenie razmerov
    f_flow>>s;							// chtenie razmerov
    f_flow>>x_grid;//z_min;								// propusk z-coordinat
    f_flow>>y_grid;//z_max;								// propusk z-coordinat
    f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;//f_trj>>s;f_trj>>s;	// chtenie musora
     //ITEM    ATOMS       id         x         y        vx        vy
}

void Flow::Read_flow_info(){
    double value;
    for(int cols = 0; cols < x_grid; cols++)
    {
        for(int row = 0; row < y_grid; row ++)
        {
            f_flow>>value;
            h_vx_grid[x_grid * row + cols] += value;
            f_flow>>value;
            h_vy_grid[x_grid * row + cols] += value;
        }
    }
}

void Flow::print_array_1D(int num_str,double *array)
{
    std::cout<<"in prind1d \n";
    for(int i = 0; i < num_str; i++)
    {
        std::cout<<array[i]<<" ";
    }
}
void Flow::print_array_2D(int num_str,int num_col,double *array)
{
    std::cout<<"in print 2d \n";
//     id_part * len_ts_for_sigm2t + id_ts
    for(int i = 0; i < num_str; i++)
    {
        for(int j = 0; j < num_col; j++)
        {
            std::cout<<"( "<<i<<" "<<array[num_col * i + j]<<") ";
        }
        std::cout<<"\n";
    }
}
void Flow::WriteGist(std::ofstream& full_name_file, int counter)
{
    double count = 0;
    for(int order = 0; order < n_el_gist; order++)
	{
		full_name_file<<h_sigma_gist[order] / ((counter - 1)  * len_ts_for_sigm2t)<<"\n";
        count += h_sigma_gist[order] / ((counter - 1) * (len_ts_for_sigm2t));
	}
	full_name_file<<count;
}
void Flow::WriteSigma2t(std::ofstream& full_name_file, int counter)
{
    cudaMemcpy(h_sigma2t_itog, d_sigma2t_itog, len_ts_for_sigm2t * sizeof(double), cudaMemcpyDeviceToHost);
    for(int order = 0; order < len_ts_for_sigm2t; order++)
	{
		full_name_file<<h_sigma2t_itog[order] / (counter - 1) <<"\n";
	}

}
void Flow::WriteImpulseMoment(std::ofstream& full_name_file, int counter)
{
    double sum = 0;
    cudaMemcpy(h_imp_of_part, d_imp_of_part, nparts * sizeof(double), cudaMemcpyDeviceToHost);
    for (int id_part = 0; id_part < nparts; id_part++)
    {
        sum += (h_imp_of_part[id_part] / nparts /counter / len_ts_for_sigm2t / 50);
    }
    full_name_file<<sum;
}
void Flow::WriteFlowEstimate(std::ofstream& full_name_file, int counter)
{
    cudaMemcpy(h_flow_estimate_per_particle, d_flow_estimate_per_particle, nparts* sizeof(double), cudaMemcpyDeviceToHost);
    double flow = 0;
    for (int id_part = 0; id_part < nparts; id_part++)
    {
        flow += (h_flow_estimate_per_particle[id_part]/nparts/ len_ts_for_sigm2t);
    }
    full_name_file<<flow;
}
void Flow::WriteAnglesInfo(std::ofstream& full_name_file1, std::ofstream& full_name_file2, int counter)
{
    for(int i = 0; i < n_elem_in_vv; i++)
    {
        full_name_file1<<h_vv_angles_r[i]<<"\n";
        full_name_file2<<h_vv_disp_angles_r[i]<<"\n";
    }
}

void Flow::WriteVelcityCorrelation(std::ofstream& full_name_file, int counter)
{
    cudaMemcpy(h_vv_cor, d_vv_cor, 4 * n_elem_in_vv * n_elem_in_vv * sizeof(double), cudaMemcpyDeviceToHost);
    for (int id_vv_x = 0; id_vv_x < 2 * n_elem_in_vv; id_vv_x++)
    {
        for(int id_vv_y = 0; id_vv_y < 2 * n_elem_in_vv; id_vv_y++)
        {
            full_name_file<< h_vv_cor[id_vv_y * 2 * n_elem_in_vv + id_vv_x] / counter<<" ";
        }
        full_name_file<<"\n";

    }
}

int main()
{

    std::string full_path, full_path2,full_save,full_save2, full_save3, full_save4, full_save5, full_save6, full_save7;
    std::string path = "/run/media/artur/Новый том/Mansevitch_quantum_dots/trj_setka/";
	std::string save_path = "/run/media/artur/Новый том/Mansevitch_quantum_dots/trj_setka/";
    std::string D_r[] = {"1"};//, "1", "1.5", "2"};
    std::string A[] = {"4"};//, "2"};
    double r_dif = 4;
    int n_in_gist = 400;
    int len_t_sigma = 64;   // только степени двойки
    int ts_in = 0;
    double r_cut_vv = 5;
    int n_elem_in_vv = 50;

    for(int ia = 0; ia < 1; ia ++)
    {
        for(int idr = 0; idr < 1; idr++)
        {
            int counter = 0;
            full_path = path + "SETKA_EXP_F_WCA_av_100_0.5_per_150_G_1_A_" + A[ia] + "_D_" +  D_r[idr] +"_rho_1.4.lammpstrj";
            full_path2 = path + "SETKA_EXP_F_WCA_av_100_0.5_per_150_G_1_A_" + A[ia] + "_D_" +  D_r[idr] +"_rho_1.4.lammpstrj_setka_";
            full_save = save_path + "GIST_" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            full_save2 = save_path + "Sigma2t_" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            full_save3 = save_path + "Imp_Moment_" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            full_save4 = save_path + "VelVel_corr_" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            full_save5 = save_path + "Flow_estimate" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            full_save6 = save_path + "Average_Angle" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            full_save7 = save_path + "Disp_Angle" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            std::cout<<full_path<<"\n";
            Flow Set(full_path, full_path2, r_dif, n_in_gist, len_t_sigma, r_cut_vv, n_elem_in_vv);
            for(int ts  = 0; ts < 600; ts++)
            {
                if(ts % len_t_sigma == 0)
                {
                    ts_in = 0;
                    counter++;
                }
                //std::cout<<"ts = "<<ts<<"\n";
                Set.Read_State();
                Set.CalcMsdGist(ts_in);
                Set.CalcImpulsMomet();
                ts_in++;
            }
            Set.CalcAnglesDispersion();

            std::ofstream fout_Gist(full_save);      // fail dl9 vuvoda dannux
            std::ofstream fout_Sigma2t(full_save2);      // fail dl9 vuvoda dannux
            std::ofstream fout_ImpMom(full_save3);      // fail dl9 vuvoda dannux
            std::ofstream fout_VvCor(full_save4);      // fail dl9 vuvoda dannux
            std::ofstream fout_FlowEstimate(full_save5);      // fail dl9 vuvoda dannux
            std::ofstream fout_Angle(full_save6);      // fail dl9 vuvoda dannux
            std::ofstream fout_DispAngle(full_save7);      // fail dl9 vuvoda dannux
            Set.WriteGist(fout_Gist, counter);
            Set.WriteSigma2t(fout_Sigma2t, counter);
            Set.WriteImpulseMoment(fout_ImpMom, counter);
            Set.WriteVelcityCorrelation(fout_VvCor, counter);
            Set.WriteFlowEstimate(fout_FlowEstimate, counter);
            Set.WriteAnglesInfo(fout_Angle,fout_DispAngle, counter);

        }
    }
    return (0);
}
