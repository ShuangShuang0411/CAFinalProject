#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <math.h>
#include <algorithm>


double Gamma = 5.0/3.0;
//int N = 1000;
//double dx = 1.0/N;
double T = 0.1;
int DataReconstruct = 1;   // Data reconstruction method: 0 for none, 1 for PPM
int NN [] = {50, 100, 200, 500, 1000, 2000};
int NThread = 2;   // Total number of threads in OpenMP


// primitive variables to conserved variables
void Conserved2Primitive ( int N, double **U, double **pri ){
    
    for (int i=0;i<N;i++){
        pri[0][i]=U[0][i];
        pri[1][i]=U[1][i]/U[0][i];
        pri[2][i]=(Gamma-1.0)*(U[2][i]-0.5*pow(U[1][i],2)/U[0][i]);
    }
}

// conserved variables to primitive variables
void Primitive2Conserved ( int N, double **pri, double **cons ){
    
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

void SoundSpeedMax( int N, double **U, double *s_max) {
    double s[N], p[N];
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

void Conserved2Flux ( int N, double **U, double **flux ){
    
    for (int i=0;i<N;i++){
        double P = ComputePressure( U[0][i], U[1][i], U[2][i]);
        double u = U[1][i] / U[0][i];
        flux[0][i] = U[1][i];
        flux[1][i] = u*U[1][i] + P;
        flux[2][i] = (P+U[2][i])*u;
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// PPM Data Reconstruciton

void get_aLR (int N, double *a, double *a_L, double *a_R, double *delm_a){

    double *a_half = new double [N];
    double *del_a  = new double [N];
//    double *delm_a = new double [N];

    for (int j=1;j<N-1;j++)   del_a[j] = 0.5*a[j+1]-0.5*a[j-1];
    del_a[0] = 0.5*a[1]-0.5*a[0];
    del_a[N-1] = 0.5*a[N-1]-0.5*a[N-2];

    for (int j=1;j<N-1;j++){
        if ( (a[j+1]-a[j])*(a[j]-a[j-1])>0 ){
            delm_a[j] = std::min( abs(del_a[j]), std::min( 2*abs(a[j]-a[j-1]), 2*abs(a[j]-a[j+1])) );  
            if ( del_a[j]<0 )  delm_a[j] *= -1.0;
        }
        else   delm_a[j] = 0.0;
    }
    delm_a[0] = 0.0;
    delm_a[N-1] = 0.0;

    for (int j=0;j<N-1;j++)   a_half[j] = 0.5*a[j]+0.5*a[j+1]+1.0/6.0*(delm_a[j]-delm_a[j+1]);
    a_half[N-1] = a[N-1];

    if ( DataReconstruct == 1 ){
        for (int j=1;j<N;j++){
            a_R[j] = a_half[j];
            a_L[j] = a_half[j-1];
        }
        a_R[0] = a_half[0];
        a_L[0] = a[0];
    }
    else if ( DataReconstruct == 2){
        for (int j=1;j<N-1;j++){
            a_L[j] = 0.5*a[j]+0.5*a[j-1]-1.0/6.0*(delm_a[j]+delm_a[j-1]);
            a_R[j] = 0.5*a[j]+0.5*a[j+1]-1.0/6.0*(delm_a[j+1]+delm_a[j]);
        }
        a_L[0] = 0.5*a[0]+0.5*a[0]-1.0/6.0*(delm_a[0]+delm_a[0]);
        a_L[N-1] = 0.5*a[N-1]+0.5*a[N-2]-1.0/6.0*(delm_a[N-1]+delm_a[N-2]); 
        a_R[0] = 0.5*a[0]+0.5*a[1]-1.0/6.0*(delm_a[1]+delm_a[0]);
        a_R[N-1] = 0.5*a[N-1]+0.5*a[N-1]-1.0/6.0*(delm_a[N-1]+delm_a[N-1]);
    }

//  Apply monotonicity constraints
    for (int j=0;j<N;j++){
        if ( (a_R[j]-a[j])*(a[j]-a_L[j])<=0 ) {
            a_L[j] = a[j];
            a_R[j] = a[j];
        }
        else if ( (a_R[j]-a_L[j])*(a[j]-0.5*a_L[j]-0.5*a_R[j]) > (a_R[j]-a_L[j])*(a_R[j]-a_L[j])/6.0 ) {
            a_L[j] = 3*a[j]-2*a_R[j];
        }
        else if ( -(a_R[j]-a_L[j])*(a_R[j]-a_L[j])/6.0 > (a_R[j]-a_L[j])*(a[j]-0.5*a_L[j]-0.5*a_R[j]) ) {
            a_R[j] = 3*a[j]-2*a_L[j];
        }
    }
}

void interpolation_fnt (int N, int LeftRight, double dx, double *y, double *a, double *value){

    double f = 0.0;
    double *x = new double [N];
    for (int j=0;j<N;j++)    x[j] = y[j]/dx;
    double *a_L = new double [N]; 
    double *a_R = new double [N]; 
    double *temp = new double [N];

    get_aLR(N, a, a_L, a_R, temp);

    for (int j=0;j<N-1;j++){
        if (LeftRight==0)   f = a_R[j]-x[j]/2*(a_R[j]-a_L[j]-(1-2/3*x[j])*6*(a[j]-0.5*a_R[j]-0.5*a_L[j]));
        else                f = a_L[j+1]+x[j]/2*(a_R[j+1]-a_L[j+1]+(1-2/3*x[j])*6*(a[j+1]-0.5*a_R[j+1]-0.5*a_L[j+1])); 
        value[j] = f;
    }
    if (LeftRight==0)  value[N-1] = a_R[N-1]-x[N-1]/2*(a_R[N-1]-a_L[N-1]-(1-2/3*x[N-1])*6*(a[N-1]-0.5*a_R[N-1]-0.5*a_L[N-1]));
    else               value[N-1] = a_L[N-1]+x[N-1]/2*(a_R[N-1]-a_L[N-1]+(1-2/3*x[N-1])*6*(a[N-1]-0.5*a_R[N-1]-0.5*a_L[N-1]));
}


void correction_HLL (int N, int LeftRight, double dt, double dx, double *lambda_p, double *lambda_n, double *lambda_0, double *cs, double **W, double **delm_W, double **value){

    double f0, f1, f2;
    double *lambda_max = new double [N];
    double *lambda_min = new double [N];

    for (int j=0;j<N;j++){
        lambda_max[j] = std::max(lambda_p[j], std::max(lambda_n[j], lambda_0[j]));
        lambda_min[j] = std::min(lambda_p[j], std::min(lambda_n[j], lambda_0[j]));
    }

    for (int j=0;j<N;j++){
        f0, f1, f2 == 0.0, 0.0, 0.0;
        if (LeftRight==0){  
            if (lambda_p[j]<0){
                f0 += (lambda_p[j]-lambda_max[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2));
                f1 += (lambda_p[j]-lambda_max[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*cs[j]/W[0][j];
                f2 += (lambda_p[j]-lambda_max[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*pow(cs[j],2);
            }
            if (lambda_n[j]<0){
                f0 += (lambda_n[j]-lambda_max[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2));
                f1 += (lambda_n[j]-lambda_max[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*(-cs[j])/W[0][j];
                f2 += (lambda_n[j]-lambda_max[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*pow(cs[j],2);
            }
            if (lambda_0[j]<0){
                f0 += (lambda_0[j]-lambda_max[j])*(delm_W[0][j]-delm_W[2][j]/pow(cs[j],2));
            }
            value[0][j] = -0.5*dt/dx*f0;
            value[1][j] = -0.5*dt/dx*f1;
            value[2][j] = -0.5*dt/dx*f2;
        }
        else {
            if (lambda_p[j]>0){
                f0 += (lambda_p[j]-lambda_min[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2));
                f1 += (lambda_p[j]-lambda_min[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*cs[j]/W[0][j];
                f2 += (lambda_p[j]-lambda_min[j])*(0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*pow(cs[j],2);
            }
            if (lambda_n[j]>0){
                f0 += (lambda_n[j]-lambda_min[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2));
                f1 += (lambda_n[j]-lambda_min[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*(-cs[j])/W[0][j];              
                f2 += (lambda_n[j]-lambda_min[j])*(-0.5*W[0][j]*delm_W[1][j]/cs[j]+0.5*delm_W[2][j]/pow(cs[j],2))*pow(cs[j],2);
            }
            if (lambda_0[j]>0){
                f0 += (lambda_0[j]-lambda_min[j])*(delm_W[0][j]-delm_W[2][j]/pow(cs[j],2));
            }
            value[0][j] = -0.5*dt/dx*f0;
            value[1][j] = -0.5*dt/dx*f1;
            value[2][j] = -0.5*dt/dx*f2;
        }
    }
}


void PPM_Hydro (int N, double dt, double dx, double **U, double **U_L, double **U_R){

    double **W = new double*[3];
    for (int i = 0; i < 3; i++)   W[i] = new double[N];
    double **W_L_prime = new double*[3];
    for (int i = 0; i < 3; i++)   W_L_prime[i] = new double[N];
    double **W_R_prime = new double*[3];
    for (int i = 0; i < 3; i++)   W_R_prime[i] = new double[N];
    double **W_L_p = new double*[3];
    for (int i = 0; i < 3; i++)   W_L_p[i] = new double[N];
    double **W_R_p = new double*[3];
    for (int i = 0; i < 3; i++)   W_R_p[i] = new double[N];
    double **W_L_n = new double*[3];
    for (int i = 0; i < 3; i++)   W_L_n[i] = new double[N];
    double **W_R_n = new double*[3];
    for (int i = 0; i < 3; i++)   W_R_n[i] = new double[N];
    double **W_L_0 = new double*[3];
    for (int i = 0; i < 3; i++)   W_L_0[i] = new double[N];
    double **W_R_0 = new double*[3];
    for (int i = 0; i < 3; i++)   W_R_0[i] = new double[N];
    double **W_L = new double*[3];
    for (int i = 0; i < 3; i++)   W_L[i] = new double[N];
    double **W_R = new double*[3];
    for (int i = 0; i < 3; i++)   W_R[i] = new double[N];
    double **delm_W = new double*[3];
    for (int i = 0; i < 3; i++)   delm_W[i] = new double[N];

    Conserved2Primitive(N, U, W);

    double *cs = new double [N];
    for (int j=0;j<N;j++){
        cs[j] = sqrt(Gamma*W[2][j]/W[0][j]);
//        if (W[1][j]>=0)  cs[j] += W[1][j];
//        else  cs[j] -= W[1][j];
    }

    double *y_L = new double [N];
    double *y_R = new double [N];
    for (int j=0;j<N;j++){
        y_L[j] = dt*(W[1][j]+cs[j]);
        if (y_L[j]<0.0)    y_L[j]=0.0;
    }
    for (int j=0;j<N-1;j++)   y_R[j] = -dt*(W[1][j+1]-cs[j+1]);
    y_R[N-1] = y_R[N-2];
    for (int j=0;j<N;j++){
        if (y_R[j]<0.0)    y_R[j]=0.0;
    }

//  1. the initial guess of W_L and W_R
    for (int i = 0; i < 3; i++){
        interpolation_fnt (N, 0, dx, y_L, W[i], W_L_prime[i]);
        interpolation_fnt (N, 1, dx, y_R, W[i], W_R_prime[i]);
    }

//  2. the eigenvalues and eigenstates of the data
    double *lambda_p = new double [N];
    double *lambda_n = new double [N];
    double *lambda_0 = new double [N];
    double *lambda_p_R = new double [N];
    double *lambda_n_R = new double [N];
    double *lambda_0_R = new double [N];

    for (int j=0;j<N;j++){
        lambda_p[j] = dt*(W[1][j] + cs[j]);
        lambda_n[j] = dt*(W[1][j] - cs[j]);
        lambda_0[j] = dt*(W[1][j]);
    }
    for (int j=0;j<N-1;j++){
        lambda_p_R[j] = -lambda_p[j+1];
        lambda_n_R[j] = -lambda_n[j+1];
        lambda_0_R[j] = -lambda_0[j+1];
    }
    lambda_p_R[N-1], lambda_n_R[N-1], lambda_0_R[N-1] = lambda_p[N-1], lambda_n[N-1], lambda_0[N-1];

    for (int i = 0; i < 3; i++){
        interpolation_fnt (N, 0, dx, lambda_p, W[i], W_L_p[i]);
        interpolation_fnt (N, 1, dx, lambda_p_R, W[i], W_R_p[i]);
        interpolation_fnt (N, 0, dx, lambda_n, W[i], W_L_n[i]);
        interpolation_fnt (N, 1, dx, lambda_n_R, W[i], W_R_n[i]);
        interpolation_fnt (N, 0, dx, lambda_0, W[i], W_L_0[i]);
        interpolation_fnt (N, 1, dx, lambda_0_R, W[i], W_R_0[i]);
    }   

//  3. Reconstruct the data
    double *C_L = new double [N];
    double *C_R = new double [N];
    double *beta_L_p = new double [N];
    double *beta_R_p = new double [N];
    double *beta_L_n = new double [N];
    double *beta_R_n = new double [N];
    double *beta_L_0 = new double [N];
    double *beta_R_0 = new double [N];

    for (int j=0;j<N;j++){
        C_L[j] = sqrt(Gamma*W_L_prime[2][j]/W_L_prime[0][j]);
        C_R[j] = sqrt(Gamma*W_R_prime[2][j]/W_R_prime[0][j]);
    }

    for (int j=0;j<N;j++){
        if (lambda_p[j]<=0)    beta_L_p[j] = 0.0;
        else                   beta_L_p[j] = -0.5/C_L[j]*(W_L_prime[1][j]-W_L_p[1][j]+(W_L_prime[2][j]-W_L_p[2][j])/C_L[j]);
        if (lambda_n[j]<=0)    beta_L_n[j] = 0.0;
        else                   beta_L_n[j] = 0.5/C_L[j]*(W_L_prime[1][j]-W_L_n[1][j]-(W_L_prime[2][j]-W_L_n[2][j])/C_L[j]);
        if (lambda_0[j]<=0)    beta_L_0[j] = 0.0;
        else                   beta_L_0[j] = (W_L_prime[2][j]-W_L_0[2][j])/C_L[j]/C_L[j]+1.0/W_L_prime[0][j]-1.0/W_L_0[0][j];
        if (lambda_p_R[j]<=0)  beta_R_p[j] = 0.0;
        else                   beta_R_p[j] = -0.5/C_R[j]*(W_R_prime[1][j]-W_R_p[1][j]+(W_R_prime[2][j]-W_R_p[2][j])/C_R[j]);
        if (lambda_n_R[j]<=0)  beta_R_n[j] = 0.0;
        else                   beta_R_n[j] = 0.5/C_R[j]*(W_R_prime[1][j]-W_R_n[1][j]-(W_R_prime[2][j]-W_R_n[2][j])/C_R[j]);
        if (lambda_0_R[j]<=0)  beta_R_0[j] = 0.0;
        else                   beta_R_0[j] = (W_R_prime[2][j]-W_R_0[2][j])/C_R[j]/C_R[j]+1.0/W_R_prime[0][j]-1.0/W_R_0[0][j];
    }  
     
    for (int j=0;j<N;j++){
        W_L[0][j] = 1.0/(1.0/(W_L_prime[0][j])-beta_L_p[j]-beta_L_n[j]-beta_L_0[j]);
        W_L[1][j] = W_L_prime[1][j] + C_L[j]*(beta_L_p[j]-beta_L_n[j]);
        W_L[2][j] = W_L_prime[2][j] + C_L[j]*C_L[j]*(beta_L_p[j]+beta_L_n[j]);
        W_R[0][j] = 1.0/(1.0/(W_R_prime[0][j])-beta_R_p[j]-beta_R_n[j]-beta_R_0[j]);
        W_R[1][j] = W_R_prime[1][j] + C_R[j]*(beta_R_p[j]-beta_R_n[j]);
        W_R[2][j] = W_R_prime[2][j] + C_R[j]*C_R[j]*(beta_R_p[j]+beta_R_n[j]);
    }

//  Correction for HLL solver 
    double **corr_L = new double*[3];
    for (int i = 0; i < 3; i++)   corr_L[i] = new double[N];
    double **corr_R = new double*[3];
    for (int i = 0; i < 3; i++)   corr_R[i] = new double[N];
    double *tempL = new double [N];
    double *tempR = new double [N];

    for (int j=0;j<N;j++){
        lambda_p[j] /= dt;
        lambda_n[j] /= dt;
        lambda_0[j] /= dt;
    }
    for (int i=0;i<3;i++)   get_aLR(N, W[i], tempL, tempR, delm_W[i]);

    correction_HLL (N, 0, dt, dx, lambda_p, lambda_n, lambda_0, cs, W, delm_W, corr_L);
    correction_HLL (N, 1, dt, dx, lambda_p, lambda_n, lambda_0, cs, W, delm_W, corr_R);

    for (int i=0;i<3;i++){
//        for (int j=0;j<N;j++)    W_L[i][j] += corr_L[i][j];
//        for (int j=0;j<N-1;j++)    W_R[i][j] += corr_R[i][j+1];
    }

    Primitive2Conserved(N, W_L, U_L);
    Primitive2Conserved(N, W_R, U_R);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////


void HLLC_Riemann_Solver (int N, double **U_L, double **U_R, double **HLLC_flux ){
    
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
    for (int i=0;i<N;i++){
        
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
        Conserved2Flux(N, U_L, F_L);
        Conserved2Flux(N, U_R, F_R);
        
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


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
int main(int argc, const char * argv[]) {
    omp_set_num_threads( NThread );

    //save data into file
    FILE * data_ptr;
    data_ptr = fopen("./bin/ac_PPM_CW_data_evol.txt", "w");
    if (data_ptr==0)  return 0;

    for (int abc=0;abc<6;abc++){  // loop over different N
    int N = NN[abc];
    double dx = 1.0/N;

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
    double **W_prime = new double*[3];
    for (int i = 0; i < 3; i++)   W_prime[i] = new double[N];
    double **W_prime_L = new double*[3];
    for (int i = 0; i < 3; i++)   W_prime_L[i] = new double[N];
    double **W_prime_R = new double*[3];
    for (int i = 0; i < 3; i++)   W_prime_R[i] = new double[N];
    double **flux_L = new double*[3];
    for (int i = 0; i < 3; i++)   flux_L[i] = new double[N];
    double **flux_R = new double*[3];
    for (int i = 0; i < 3; i++)   flux_R[i] = new double[N];
    double *temp = new double [N];


    double dt; 
    double t = 0.; 
    double S_max = 6.29;

    //set the initial condition
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
/*
    for (int i=0;i<3;i++){
        for (int j=0;j<N;j++){
            fprintf(data_ptr,"%e ", W[i][j]);
        }
        fprintf(data_ptr,"\n");
    }
*/    
    Primitive2Conserved(N, W, U);
    
    while (t<=T){

        //compute dt
        SoundSpeedMax(N, U, &S_max);
        dt = dx/S_max;
        printf("Debug: dt = %.10f, t = %.10f\n", dt, t);
        t += dt; 

        if (DataReconstruct == 0) {        
            for (int i=0;i<N-1;i++){
                U_R[0][i] = U[0][i+1];
                U_R[1][i] = U[1][i+1];
                U_R[2][i] = U[2][i+1];
            }
            U_R[0][N-1] = U[0][N-1];
            U_R[1][N-1] = U[1][N-1];
            U_R[2][N-1] = U[2][N-1];
            for (int i=1;i<N;i++){
                U_L[0][i] = U[0][i-1];
                U_L[1][i] = U[1][i-1];
                U_L[2][i] = U[2][i-1];
            }
            U_L[0][0] = U[0][0];
            U_L[1][0] = U[1][0];
            U_L[2][0] = U[2][0];
        }
        else if (DataReconstruct == 1) {  // PPM data reconstruction
            PPM_Hydro(N, dt, dx, U, U_L, U_R);
/*
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
*/
        }
        else if (DataReconstruct == 2) {  // PPM data reconstruction test
            Conserved2Primitive(N, U, W_prime);
            for (int i=0;i<3;i++){
                get_aLR(N, W_prime[i], W_prime_L[i], W_prime_R[i], temp);
                for (int j=0;j<N;j++)   W_prime_L[i][j] = 6*W_prime[i][j]-2*W_prime_L[i][j]-3*W_prime_R[i][j];
            }
            Primitive2Conserved(N, W_prime_L, U_L);
            Primitive2Conserved(N, W_prime_R, U_R);

            Conserved2Flux(N, U_L, flux_L);
            Conserved2Flux(N, U_R, flux_R);

            for (int i=1;i<N-1;i++){
                U_L[0][i] -= (flux_R[0][i]-flux_L[0][i])*0.5*dt/dx;
                U_L[1][i] -= (flux_R[1][i]-flux_L[1][i])*0.5*dt/dx;
                U_L[2][i] -= (flux_R[2][i]-flux_L[2][i])*0.5*dt/dx;
                U_R[0][i] -= (flux_R[0][i]-flux_L[0][i])*0.5*dt/dx;
                U_R[1][i] -= (flux_R[1][i]-flux_L[1][i])*0.5*dt/dx;
                U_R[2][i] -= (flux_R[2][i]-flux_L[2][i])*0.5*dt/dx;
            } 
            
        }

//        printf("Debug: left rho = %.10f, right rho = %10f\n");
        //update data
/*
        HLLC_Riemann_Solver(U,U_R,HLLC_flux_R);
        HLLC_Riemann_Solver(U_L,U,HLLC_flux_L);

        for (int i=1;i<N-1;i++){
            U[0][i] -= (HLLC_flux_R[0][i]-HLLC_flux_L[0][i])*dt/dx;
            U[1][i] -= (HLLC_flux_R[1][i]-HLLC_flux_L[1][i])*dt/dx;
            U[2][i] -= (HLLC_flux_R[2][i]-HLLC_flux_L[2][i])*dt/dx;
        }
*/
//      With data reconstruction
        HLLC_Riemann_Solver(N, U_L, U_R, HLLC_flux_R);
        for (int i=1;i<N-1;i++){
            U[0][i] -= (HLLC_flux_R[0][i]-HLLC_flux_R[0][i-1])*dt/dx; 
            U[1][i] -= (HLLC_flux_R[1][i]-HLLC_flux_R[1][i-1])*dt/dx; 
            U[2][i] -= (HLLC_flux_R[2][i]-HLLC_flux_R[2][i-1])*dt/dx; 
        }

        //boundary condition: outflow
        U[0][0] = U[0][1];
        U[1][0] = U[1][1];
        U[2][0] = U[2][1];
        U[0][N-1] = U[0][N-2];
        U[1][N-1] = U[1][N-2];
        U[2][N-1] = U[2][N-2];
    }
    
    Conserved2Primitive(N, U, W);

    
    //save data into file
    for (int i=0;i<3;i++){
        for (int j=0;j<N;j++){
            fprintf(data_ptr,"%e ", W[i][j]);
        }
        for (int j=N;j<2000;j++)  fprintf(data_ptr,"%e ", 0.0);
        fprintf(data_ptr,"\n");
    }

    delete[] U;
    delete[] W;
    delete[] HLLC_flux_L;
    delete[] HLLC_flux_R;
    delete[] U_L;
    delete[] U_R;

    } // loop over different N   

    fclose(data_ptr);
    
    return 0;
}

