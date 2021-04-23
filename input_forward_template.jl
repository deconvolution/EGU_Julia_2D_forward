## import packages
using MAT,Plots,Dates,TimerOutputs,WriteVTK,DataFrames,CSV

const USE_GPU=false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end
include("./seismic2D_function.jl");
Threads.nthreads()
## timing
ti=TimerOutput();
## define model parameters
nx=200;
nz=200;
dt=.001;
dx=10;
dz=10;
nt=2000;

X=(1:nx)*dx;
Z=(nz:-1:1)*dz;

A=@ones(nx,nz)*2000;
A[:,100:end] .=3000;

#vp[:,1:29] .=340;
#vp[:,end-29:end] .=0;

# PML layers
lp=20;

# PML coefficient, usually 2
nPML=2;

# Theoretical coefficient, more PML layers, less R
# Empirical values
# lp=[10,20,30,40]
# R=[.1,.01,.001,.0001]
Rc=.0001;

# generate empty density
rho=@ones(nx,nz)*1;

# Lame constants for solid

lambda=rho.*A.^2;
mu=copy(lambda)*1;

rho=@ones(nx,nz)*1;
## assign stiffness matrix and rho
mutable struct C2
    C11
    C13
    C15
    C33
    C35
    C55
    rho
end
C=C2(lambda+2*mu,lambda,@zeros(nx,nz),lambda+2*mu,@zeros(nx,nz),mu,rho);
## source
# source location
# multiple source at one time or one source at each time
msot=0;

# source location grid
s_s1=zeros(Int32,1,1);
s_s3=copy(s_s1);
s_s1[:] .=100;
s_s3[:] .=30;

# source locations true
s_s1t=dx .*s_s1;
s_s3t=maximum(Z) .-dz .*s_s3;

# magnitude
M=2.7;
# source frequency [Hz]
freq=5;

# source signal
singles=rickerWave(freq,dt,nt,M);

# give source signal to x direction
s_src1=zeros(Float32,nt,1);
s_src1=0*repeat(singles,1,length(s_s3));

# give source signal to z direction
s_src3=copy(s_src1);
s_src3=1*repeat(singles,1,length(s_s3));

# source type. 'D' for directional source. 'P' for P-source.
s_source_type=["D"];
## receiver
# receiver locations grid
r1=ones(Int32,1,15);
r1[:] =30:10:nx-30;

r3=ones(Int32,1,15);
r3[:] = ones(1,15)*30;

# receiver locations true
r1t=dx .*r1;
r3t=maximum(Z) .- dz .*r3;

## plot
# point interval in time steps, 0 = no plot
plot_interval=100;
# save wavefield
wavefield_interval=0;
## create folder for saving
p2= @__FILE__;
p3=chop(p2,head=0,tail=3);
if isdir(p3)==0
    mkdir(p3);
end
## mute some receiver components
Rm=ones(nt,length(r3),3);
Rm[:,:,3] .=0;
## initialize seismograms
data=zeros(nt,length(r3));
## implement solver
implement_2D_forward(dt,dx,dz,nt,nx,nz,
    X,Z,
    r1,r3,
    Rm,
    s_s1,s_s3,s_src1,s_src3,s_source_type,
    r1t,r3t,
    s_s1t,s_s3t,
    lp,nPML,Rc,
    C,
    plot_interval,
    wavefield_interval,
    p3);
## plot seismograms
file=matopen(string(path_rec,"/rec_p.mat"));
tt=read(file,"data");
close(file);
ir=1;
plot(dt:dt:dt*nt,tt[:,ir])
