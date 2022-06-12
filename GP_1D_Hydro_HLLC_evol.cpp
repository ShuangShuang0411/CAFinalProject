#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>

#include "Eigen/LU"
using Eigen::MatrixXd;
using Eigen::VectorXd;


float Gamma = 5.0/3.0;
int N_in = 256;
int R = 2; //stencil radii for Gaussian process
int nghost = R + 1; // number of ghost zones
int N = N_in + 2 * nghost;
double dx = 1.0/N_in;

double l = 30; //hyperparameter for SE kernel (the best value depends on the problem) 

int NThread = 4;   // Total number of threads in OpenMP


double T = 0.1;

//initial condition: Sod's shock tube
void InitialCondition_Sod_shock_tube ( double **W ){
    for (int i=0;i<N;i++){
        if (i<N/2-1){
            W[0][i] = 1.0;
            W[1][i] = 0.0;
            W[2][i] = 1.0;
        }
        else{
            W[0][i] = 0.125;
            W[1][i] = 0.0;
            W[2][i] = 0.1;
        }
    }
}

//initial condition: acoustic wave
void InitialCondition_acoustic_wave ( double **W, double time=0){
    double L     = 1;
    double cs    = 1.0;      // sound speed
    double d_amp = 1.0e-5;    // density perturbation amplitude
    double d0    = 1.0;       // density background
    double u1 = cs*d_amp/d0;        // velocity perturbation
    double P0 = pow(cs, 2.0)*d0/Gamma;   // background pressure
    double P1 = pow(cs, 2.0)*d_amp;      // pressure perturbation
    for (int i=0;i<N;i++){
        W[0][i] = d0 + d_amp*sin(2.0*M_PI*i*L/(N_in) + 2.0*M_PI*time);
        W[1][i] = u1*sin(2.0*M_PI*i*L/(N_in) + 2.0*M_PI*time);
        W[2][i] = P0 + P1*sin(2.0*M_PI*i*L/(N_in) + 2.0*M_PI*time);
    }
}

//boundary condition: outflow
void BoundaryCondition_outflow ( double **U ){
    for (int i=0;i<nghost;i++){
        U[0][i] = U[0][nghost];
        U[1][i] = U[1][nghost];
        U[2][i] = U[2][nghost];
        U[0][N-1-i] = U[0][N-1-nghost];
        U[1][N-1-i] = U[1][N-1-nghost];
        U[2][N-1-i] = U[2][N-1-nghost];
    }
}
//boundary condition: periodic
void BoundaryCondition_periodic ( double **U ){
    double **Copy = new double*[3];
    for (int i = 0; i < 3; i++)   Copy[i] = new double[2*nghost];
    for (int i=0; i<nghost; i++){
        Copy[0][i] = U[0][i+N_in];
        Copy[1][i] = U[1][i+N_in];
        Copy[2][i] = U[2][i+N_in];}
    for (int i=nghost; i<2*nghost; i++){
        Copy[0][i] = U[0][i];
        Copy[1][i] = U[1][i];
        Copy[2][i] = U[2][i];}        
    for (int i=0;i<nghost;i++){
        U[0][i] = Copy[0][i];
        U[1][i] = Copy[1][i];
        U[2][i] = Copy[2][i];
        U[0][N-1-i] = Copy[0][2*nghost-i-1];
        U[1][N-1-i] = Copy[1][2*nghost-i-1];
        U[2][N-1-i] = Copy[2][2*nghost-i-1];
    }
}

// primitive variables to conserved variables
void Conserved2Primitive ( double **U, double **pri ){

#   pragma omp parallel for
    for (int i=0;i<N;i++){
        pri[0][i]=U[0][i];
        pri[1][i]=U[1][i]/U[0][i];
        pri[2][i]=(Gamma-1.0)*(U[2][i]-0.5*pow(U[1][i],2)/U[0][i]);
    }
}

// conserved variables to primitive variables
void Primitive2Conserved ( double **pri, double **cons ){

#   pragma omp parallel for    
    for (int i=0;i<N;i++){
        cons[0][i]=pri[0][i];
        cons[1][i]=pri[1][i]*pri[0][i];
        cons[2][i]=pri[2][i]/(Gamma-1)+0.5*pri[0][i]*pow(pri[1][i],2);
    }
}

double ComputePressure( double tho, double px, double e ){
    double p = (Gamma-1.0)*( e - 0.5*pow(px,2)/tho);
    return p;
}

void SoundSpeedMax( double **U, double *s_max) {
    double s[N], p[N];

#   pragma omp parallel for
    for (int i=0;i<N;i++){
        p[i] = ComputePressure(U[0][i],U[1][i],U[2][i]);
        s[i] = sqrt(Gamma*p[i]/U[0][i]);
        if (U[1][i]/U[0][i]>=0)  s[i] += U[1][i]/U[0][i];
        else  s[i] -= U[1][i]/U[0][i];
    }
    *s_max = 0.;
    for (int i=0;i<N;i++){
        if (s[i]>*s_max)  *s_max = s[i];
    }
}

void Conserved2Flux ( double **U, double **flux ){

#   pragma omp parallel for    
    for (int i=0;i<N;i++){
        double P = ComputePressure( U[0][i], U[1][i], U[2][i]);
        double u = U[1][i] / U[0][i];
        flux[0][i] = U[1][i];
        flux[1][i] = u*U[1][i] + P;
        flux[2][i] = (P+U[2][i])*u;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
// Gaussian Process data reconstruction [Ref. Reyes2018] (w/o WENO)
// Squared Exponential function
double SE(double x, double y){
    double r = abs(x - y);
    double K = exp(-pow(r/l, 2)/2);
    return K;
}

// calculated from covariance kernels
void Weights_vector( double *weights, double pos){
    int kernel_length = 1+2*R;
    MatrixXd K(kernel_length, kernel_length);
    VectorXd k(kernel_length), w(kernel_length);

    for (int i = 0; i < kernel_length; i++) {
        k(i) = SE(i-R, pos);
        // printf( "k[%d]: %0.3f\n", i, k(i));
        for (int j = 0; j < kernel_length; j++) {
            K(i, j) = SE(i, j);
        }
    }

    // std::cout << "Here is the matrix K:" << std::endl << K << std::endl;
    // std::cout << "Its inverse is:" << std::endl << K.inverse() << std::endl;

    w = k.transpose() * K.inverse();
    for (int i = 0; i < kernel_length; i++) {
        weights[i] = w(i);
        // printf( "w[%d]: %0.3f\n", i, w(i));
    }
}

void DataReconstruction_GP( double **U_L, double **U_R, double **U, double **W, double *w_L, double *w_R ){
    double **W_L = new double*[3];
    for (int i = 0; i < 3; i++)   W_L[i] = new double[N];
    double **W_R = new double*[3];
    for (int i = 0; i < 3; i++)   W_R[i] = new double[N];

    Conserved2Primitive(U, W);

    #   pragma omp parallel for
    for (int i = nghost; i < N-nghost; i++){
    for (int j=0; j<3; j++){
        W_L[j][i] = 0.0;
        W_R[j][i] = 0.0;
        for (int k = 0; k < 1+2*R; k++){
            W_L[j][i] += W[j][i+k-R]*w_L[k];
            W_R[j][i] += W[j][i+k-R]*w_R[k];  
            // printf( "weights_L[%d]: %0.3f\n", k, w_L[k]);
            // printf( "W[%d][%d]: %e\n", j, i+k-R, W[j][i+k-R]);
            // printf( "W_L[%d][%d]: %e\n", j, i, W_L[j][i]);
        }
    }   
    }
    // printf("%e\n", W_L[1][5]);
 
    BoundaryCondition_periodic(W_L);
    BoundaryCondition_periodic(W_R);

    Primitive2Conserved(W_L, U_L);
    Primitive2Conserved(W_R, U_R); 

    // BoundaryCondition_periodic(U_L);  
    // BoundaryCondition_periodic(U_R);  
}

//////////////////////////////////////////////////////////////////////////////////////////


void HLLC_Riemann_Solver ( double **U_L, double **U_R, double **HLLC_flux ){
    
    double **F_L = new double*[3];
    for (int i = 0; i < 3; i++)   F_L[i] = new double[N]; 
    double **F_R = new double*[3];
    for (int i = 0; i < 3; i++)   F_R[i] = new double[N]; 
    double **F_star_L = new double*[3];
    for (int i = 0; i < 3; i++)   F_star_L[i] = new double[N]; 
    double **F_star_R = new double*[3];
    for (int i = 0; i < 3; i++)   F_star_R[i] = new double[N]; 
    
    double a_L[N], a_R[N];
    double u_L[N], u_R[N];
    double p_R[N], p_L[N], p_star[N];
    double q_L[N], q_R[N], S_L[N], S_R[N], S_star[N];

#   pragma omp parallel for    
    for (int i=nghost;i<N-nghost+1;i++){
        
        u_L[i] = U_L[1][i]/U_L[0][i];
        u_R[i] = U_R[1][i]/U_R[0][i];
        p_L[i] = ComputePressure(U_L[0][i],U_L[1][i],U_L[2][i]);
        p_R[i] = ComputePressure(U_R[0][i],U_R[1][i],U_R[2][i]);
        a_L[i] = sqrt(Gamma*p_L[i]/U_L[0][i]);
        a_R[i] = sqrt(Gamma*p_R[i]/U_R[0][i]);
        
        //step 1: pressure estimate
        p_star[i] = 0.5*(p_L[i]+p_R[i])-0.5*(u_R[i]-u_L[i])*0.5*(U_L[0][i]+U_R[0][i])*0.5*(a_L[i]+a_R[i]);
        if (p_star[i]<0)   p_star[i] = 0.;
        
        //step 2: wave speed estimate
        if (p_star[i]>p_L[i]){
            q_L[i] = sqrt(1+(Gamma+1)*(p_star[i]/p_L[i]-1)/2.0/Gamma);
        }
        else  q_L[i]=1.0;
        if (p_star[i]>p_R[i]){
            q_R[i] = sqrt(1+(Gamma+1)*(p_star[i]/p_R[i]-1)/2.0/Gamma);
        }
        else  q_R[i]=1.0;
        
        S_L[i] = u_L[i]-a_L[i]*q_L[i];
        S_R[i] = u_R[i]+a_R[i]*q_R[i];
        S_star[i] = (p_R[i]-p_L[i]+U_L[1][i]*(S_L[i]-u_L[i])-U_R[1][i]*(S_R[i]-u_R[i]))/(U_L[0][i]*(S_L[i]-u_L[i])-U_R[0][i]*(S_R[i]-u_R[i]));
        
        //step 3: HLLC flux
        Conserved2Flux(U_L, F_L);
        Conserved2Flux(U_R, F_R);
        
        F_star_L[0][i] = S_star[i]*(S_L[i]*U_L[0][i]-F_L[0][i])/(S_L[i]-S_star[i]);
        F_star_L[1][i] = (S_star[i]*(S_L[i]*U_L[1][i]-F_L[1][i])+S_L[i]*(p_L[i]+U_L[0][i]*(S_L[i]-u_L[i])*(S_star[i]-u_L[i])))/(S_L[i]-S_star[i]);
        F_star_L[2][i] = (S_star[i]*(S_L[i]*U_L[2][i]-F_L[2][i])+S_L[i]*S_star[i]*(p_L[i]+U_L[0][i]*(S_L[i]-u_L[i])*(S_star[i]-u_L[i])))/(S_L[i]-S_star[i]);
        F_star_R[0][i] = S_star[i]*(S_R[i]*U_R[0][i]-F_R[0][i])/(S_R[i]-S_star[i]);
        F_star_R[1][i] = (S_star[i]*(S_R[i]*U_R[1][i]-F_R[1][i])+S_R[i]*(p_R[i]+U_L[0][i]*(S_R[i]-u_R[i])*(S_star[i]-u_R[i])))/(S_R[i]-S_star[i]);
        F_star_R[2][i] = (S_star[i]*(S_R[i]*U_R[2][i]-F_R[2][i])+S_R[i]*S_star[i]*(p_R[i]+U_L[0][i]*(S_R[i]-u_R[i])*(S_star[i]-u_R[i])))/(S_R[i]-S_star[i]);
        
        if (S_L[i]>=0){
            HLLC_flux[0][i] = F_L[0][i];
            HLLC_flux[1][i] = F_L[1][i];
            HLLC_flux[2][i] = F_L[2][i];
        }
        else if (S_L[i]<=0 && S_star[i]>=0){
            HLLC_flux[0][i] = F_star_L[0][i];
            HLLC_flux[1][i] = F_star_L[1][i];
            HLLC_flux[2][i] = F_star_L[2][i];
        }
        else if (S_star[i]<=0 && S_R[i]>=0){
            HLLC_flux[0][i] = F_star_R[0][i];
            HLLC_flux[1][i] = F_star_R[1][i];
            HLLC_flux[2][i] = F_star_R[2][i];
        }
        else if (S_R[i]<=0){
            HLLC_flux[0][i] = F_R[0][i];
            HLLC_flux[1][i] = F_R[1][i];
            HLLC_flux[2][i] = F_R[2][i];
        }
    }
}


int main(int argc, const char * argv[]) {

//  OpenMP: Set the number of threads
    omp_set_num_threads( NThread );

    double start;
    double end;
    start = omp_get_wtime(); 

    double **U = new double*[3];
    for (int i = 0; i < 3; i++)   U[i] = new double[N]; 
    double **W = new double*[3];
    for (int i = 0; i < 3; i++)   W[i] = new double[N];
    double **HLLC_flux_L = new double*[3];
    for (int i = 0; i < 3; i++)   HLLC_flux_L[i] = new double[N];
    double **HLLC_flux_R = new double*[3];
    for (int i = 0; i < 3; i++)   HLLC_flux_R[i] = new double[N];
    double **U_L = new double*[3];
    for (int i = 0; i < 3; i++)   U_L[i] = new double[N];
    double **U_R = new double*[3];
    for (int i = 0; i < 3; i++)   U_R[i] = new double[N];
    double **flux_L = new double*[3];
    for (int i = 0; i < 3; i++)   flux_L[i] = new double[N];
    double **flux_R = new double*[3];
    for (int i = 0; i < 3; i++)   flux_R[i] = new double[N];

    double **W_init = new double*[3];
    for (int i = 0; i < 3; i++)   W_init[i] = new double[N];  
    
    int num = 0;
    double dt;
    double t = 0.;
    double S_max = 6.29;

    double weights_L[1+2*R];
    double weights_R[1+2*R];

    Weights_vector(weights_L, -0.5);
    Weights_vector(weights_R, 0.5);
    
    //save data into file
    FILE * data_ptr;
    data_ptr = fopen("./bin/data_evol.txt", "w");
    if (data_ptr==0)  return 0;
    
    //set the initial condition
    InitialCondition_acoustic_wave(W);
    BoundaryCondition_periodic(W); 

    
    for (int i=0;i<3;i++){
        for (int j=0;j<N;j++){
            fprintf(data_ptr,"%e ", W[i][j]);
        }
        fprintf(data_ptr,"\n");
    }
    Primitive2Conserved(W, U);

    
    while (t+dt<=T){


//      Compute dt
        SoundSpeedMax(U, &S_max);
        dt = dx/S_max;
        t += dt; 
        printf("Debug: dt = %.10f, t = %.10f\n", dt, t);
        num += 1;

//      MUSCL-Hancock scheme step 1: Data reconstruction

        // data reconstruction using (smooth) Gaussian process 
        DataReconstruction_GP(U_L, U_R, U, W, weights_L, weights_R);
        

//      MUSCL-Hancock scheme step 2: Evolve the face-centered data by dt/2
        Conserved2Flux(U_L, flux_L);
        Conserved2Flux(U_R, flux_R);

        for (int i=1;i<N-1;i++){
            U_L[0][i] -= (flux_R[0][i]-flux_L[0][i])*0.5*dt/dx;
            U_L[1][i] -= (flux_R[1][i]-flux_L[1][i])*0.5*dt/dx;
            U_L[2][i] -= (flux_R[2][i]-flux_L[2][i])*0.5*dt/dx;
            U_R[0][i] -= (flux_R[0][i]-flux_L[0][i])*0.5*dt/dx;
            U_R[1][i] -= (flux_R[1][i]-flux_L[1][i])*0.5*dt/dx;
            U_R[2][i] -= (flux_R[2][i]-flux_L[2][i])*0.5*dt/dx;
        }  
        BoundaryCondition_periodic(U_L); 
        BoundaryCondition_periodic(U_R); 
        
        for (int i=N-1;i>0;i--){
            U_R[0][i] = U_R[0][i-1];
            U_R[1][i] = U_R[1][i-1];
            U_R[2][i] = U_R[2][i-1];
        }
        BoundaryCondition_periodic(U_R); 

//      MUSCL-Hancock scheme step 3: Riemann solver (solve the flux at the left interface)
        HLLC_Riemann_Solver(U_R,U_L,HLLC_flux_L);

//      MUSCL-Hancock scheme step 4: Evolve the volume-averaged data by dt
        for (int i=nghost;i<N-nghost;i++){
            U[0][i] -= (HLLC_flux_L[0][i+1]-HLLC_flux_L[0][i])*dt/dx;
            U[1][i] -= (HLLC_flux_L[1][i+1]-HLLC_flux_L[1][i])*dt/dx;
            U[2][i] -= (HLLC_flux_L[2][i+1]-HLLC_flux_L[2][i])*dt/dx;
        }
 
  

        BoundaryCondition_periodic(U); 


        // break;  //debug
    }
    
    
    Conserved2Primitive(U, W);
    
    end = omp_get_wtime(); 

    InitialCondition_acoustic_wave(W_init, t);
    BoundaryCondition_periodic(W_init); 
    double error = 0;
    for (int i=nghost;i<N-nghost;i++){
        error += fabs(W_init[0][i] - W[0][i]);
    }
    error /= N_in;

    printf("N = %d, total threads = %d\n", N, NThread);
    printf("Wall-clock time = %6f, number of iteration = %d\n", end-start, num);    
    printf("Errors = %e\n", error);    
    //save data into file
    for (int i=0;i<3;i++){
        for (int j=0;j<N;j++){
            fprintf(data_ptr,"%e ", W[i][j]);
        }
        fprintf(data_ptr,"\n");
    }

    fclose(data_ptr);
    
    delete[] U;
    delete[] W;
    delete[] HLLC_flux_L;
    delete[] HLLC_flux_R;
    delete[] U_L;
    delete[] U_R;
    
    return 0;
}



