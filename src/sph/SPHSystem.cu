// SPH-DualGPU
// SPH system base class implementation source code

#include "SPHSystem.cuh"


void SPHSystem::initialize(float radius,vector3 *pos, int n) {
  m_pos = pos;
  m_n = n;

  vector3* vel = new vector3[n];
  vector3* acc = new vector3[n];

  // create a ThreadManager to create a management pool for data sharing between 2 GPUs
  InteractionManager* tm = new InteractionManager();


  // initialize two threads for Dual GPU computation
  kt = new KinematicTread(tm);
  kt->kInitialize(radius, pos, vel, acc, n);
  dt = new DynamicThread(tm);
  dt->dInitialize(radius, pos, vel, acc, n);
}

void SPHSystem::doStepDynamics(float time_step) {
  
  kt->doKinematicStep();
  dt->doDynamicStep();
}

void SPHSystem::printCSV(std::string filename) {
  // create file
  std::ofstream csvFile(filename);

  csvFile << "x_pos,y_pos,z_pos" << std::endl;

  // write particle data into csv file
  for (int i = 0; i < m_n; i++) {
    csvFile << m_pos[i].x << "," << m_pos[i].y << "," << m_pos[i].z
            << std::endl;
  }

  csvFile.close();
}

// *----------------------------------------
// Kinematic kernals
__global__ void kinematicTestKernel(){
    printf("kinematic run on GPU \n");
  
}

__global__ void kinematic1stPass(vector3* pos, int n, float tolerance,float radius, int* res_arr){
  int idx = threadIdx.x;
  int count = 0;  // count total number of valid contact for the current particle
  if(idx > n){return;}

  for(int i = n-1; i > idx; i--)
  {
    float dist2 = (pos[idx].x - pos[i].x)*(pos[idx].x - pos[i].x)+
    (pos[idx].y - pos[i].y)*(pos[idx].y - pos[i].y)+(pos[idx].z - pos[i].z)*(pos[idx].z - pos[i].z);

    if (dist2 < (radius*2 + tolerance) * (radius*2 + tolerance))
    {
      count++;
    }
  }
  res_arr[idx] = count;
  __syncthreads();
}

__global__ void kinematic2ndPass(vector3* pos, int n, int* offset,int* contact_num_arr,  
                                float tolerance,float radius, contactData* pair_data){
  int idx = threadIdx.x;
  if(contact_num_arr[idx]!=0){
    
    int cur_idx = offset[idx];
    
    for(int i = n-1; i > idx; i--)
    {
      float dist2 = (pos[idx].x - pos[i].x)*(pos[idx].x - pos[i].x)+
      (pos[idx].y - pos[i].y)*(pos[idx].y - pos[i].y)+(pos[idx].z - pos[i].z)*(pos[idx].z - pos[i].z);

      if (dist2 < (radius*2 + tolerance) * (radius*2 + tolerance))
      {
        pair_data[cur_idx].contact_pair.x = idx;
        pair_data[cur_idx].contact_pair.y = i;
        cur_idx++;
      }
    }
  }

  __syncthreads();
  
}

// END of Kinematic kernels
// *----------------------------------------


void KinematicTread::kInitialize(float radius,vector3* pos, vector3* vel, vector3* acc, int n)
{
  k_n = n;
  k_radius = radius;
  cudaSetDevice(this->device_id);
  cudaMalloc(&k_pos, k_n*sizeof(vector3));
  cudaMalloc(&k_vel, k_n*sizeof(vector3));
  cudaMalloc(&k_acc, k_n*sizeof(vector3));

  cudaMemcpy(k_pos, pos, k_n*sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(k_vel, vel, k_n*sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(k_acc, acc, k_n*sizeof(vector3), cudaMemcpyHostToDevice);
}

void KinematicTread::doKinematicStep()
{
  float tolerance = 0.05;
  cudaSetDevice(this->device_id);
  //kinematicTestKernel<<<1, 1, 0, kStream>>>();

  // for each step, the kinematic thread needs to do two passes
  // first pass - look for 'number' of potential contacts
  // crate an array to store number of valid potential contacts
  int* cpu_num_arr = new int[k_n];
  for(int i = 0; i<k_n;i++)
  {
    cpu_num_arr[i] = -1;
  }
  int* k_num_arr;
  cudaMalloc(&k_num_arr, k_n*sizeof(int));
  cudaMemcpy(k_num_arr, cpu_num_arr, k_n*sizeof(int), cudaMemcpyHostToDevice);

  // first kinematic pass to calculate offset array
  kinematic1stPass<<<1,k_n,0,kStream>>>(k_pos, k_n, tolerance, k_radius, k_num_arr);

  cudaDeviceSynchronize();

  cudaMemcpy(cpu_num_arr, k_num_arr, k_n*sizeof(int), cudaMemcpyDeviceToHost);

  // calculate the offset array
  int cur_idx = 0;
  int* offset_arr = new int[k_n];
  for(int i = 0; i<k_n;i++)
  {
    offset_arr[i] = cur_idx;
    cur_idx = cur_idx + cpu_num_arr[i];
  }

  int* gpu_offset_arr;
  cudaMalloc(&gpu_offset_arr, k_n*sizeof(int));
  cudaMemcpy(gpu_offset_arr, offset_arr, k_n*sizeof(int), cudaMemcpyHostToDevice);

  // calculate total number of contact
  int contact_sum = 0;

  for(int i = 0; i < k_n; i++)
  {
    contact_sum = contact_sum + cpu_num_arr[i];
  }

  // second kinematic pass to fill the contact pair array
  contactData* cpu_pair_data = new contactData[contact_sum];
  contactData* gpu_pair_data;
  cudaMalloc(&gpu_pair_data, contact_sum*sizeof(contactData));
  cudaMemcpy(gpu_pair_data, cpu_pair_data, contact_sum*sizeof(contactData), cudaMemcpyHostToDevice);

  kinematic2ndPass<<<1,k_n,0,kStream>>>(k_pos,k_n,gpu_offset_arr,k_num_arr,tolerance,k_radius,gpu_pair_data);

  cudaMemcpy(cpu_pair_data, gpu_pair_data, contact_sum*sizeof(contactData), cudaMemcpyDeviceToHost);

  //std::cout<<"contact pair num: "<<contact_sum<<std::endl;
  //std::cout<<"i: "<<cpu_pair_data[0].contact_pair.x<<", j:"<<cpu_pair_data[0].contact_pair.y<<std::endl;


  // share data through a common ThreadManager instance
  threadManager->contact_pair = cpu_pair_data;
  threadManager->contact_pair_n = contact_sum;

  threadManager->offset = offset_arr;
  threadManager->offset_n = k_n;

  cudaDeviceSynchronize();
}

// *----------------------------------------
// Dynamic kernals
__global__ void dynamicTestKernel(){
    printf("dynamic run on GPU \n");
}

__global__ void dynamicPass(contactData* gpu_pair_data, int gpu_pair_n, vector3* gpu_pos, vector3* gpu_vel, vector3* gpu_acc, float radius){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= gpu_pair_n){return;}

  float dir_x = gpu_pos[gpu_pair_data[idx].contact_pair.x].x - gpu_pos[gpu_pair_data[idx].contact_pair.y].x;
  float dir_y = gpu_pos[gpu_pair_data[idx].contact_pair.x].y - gpu_pos[gpu_pair_data[idx].contact_pair.y].y;
  float dir_z = gpu_pos[gpu_pair_data[idx].contact_pair.x].z - gpu_pos[gpu_pair_data[idx].contact_pair.y].z;

  float dist2 = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;
  if(dist2 < radius * radius)
  {
    // TODO
  }


  __syncthreads();
}



// END of Dynamic kernels
// *----------------------------------------


void DynamicThread::dInitialize(float radius,vector3* pos, vector3* vel, vector3* acc, int n)
{
  d_n = n;
  d_radius = radius;
  cudaSetDevice(this->device_id);
  cudaMalloc(&d_pos, d_n*sizeof(vector3));
  cudaMalloc(&d_vel, d_n*sizeof(vector3));
  cudaMalloc(&d_acc, d_n*sizeof(vector3));

  cudaMemcpy(d_pos, pos, d_n*sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vel, vel, d_n*sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_acc, acc, d_n*sizeof(vector3), cudaMemcpyHostToDevice);
}

void DynamicThread::doDynamicStep()
{
  cudaSetDevice(this->device_id);
  //dynamicTestKernel<<<1, 1, 0, dStream>>>();

  // retrieve contact pair data from the ThreadManager
  contactData* cpu_pair = threadManager->contact_pair;
  int cpu_pair_n = threadManager->contact_pair_n;
  
  int* cpu_offset = threadManager->offset;
  int cpu_offset_n = threadManager->offset_n;

  // copy data to the Dynamic GPU
  contactData* gpu_pair_data;
  cudaMalloc(&gpu_pair_data, cpu_pair_n*sizeof(contactData));
  cudaMemcpy(gpu_pair_data, cpu_pair, cpu_pair_n*sizeof(contactData), cudaMemcpyHostToDevice);

  // calculate number of threads needed and number of block needed
  int num_thread = 64;
  int num_block = cpu_pair_n / num_thread + 1;

  dynamicPass<<<num_block, num_thread, 0, dStream>>>(gpu_pair_data, cpu_pair_n, d_pos, d_vel, d_acc, d_radius);

  cudaDeviceSynchronize();
}