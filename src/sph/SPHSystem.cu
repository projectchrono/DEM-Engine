// SPH-DualGPU
// SPH system base class implementation source code

#include <sph/SPHSystem.cuh>
#include <kernel/cudaKernels.cuh>


void SPHSystem::initialize(float radius,vector3 *pos, int n) {

  shared_data = new ExchangeData();

  m_pos = pos;
  m_n = n;

  vector3* vel = new vector3[n];
  vector3* acc = new vector3[n];

  GpuManager* gm = new GpuManager(2);


  // initialize two threads for Dual GPU computation
  kt = new KinematicTread(gm,shared_data);
  kt->kInitialize(radius, pos, vel, acc, n);
  dt = new DynamicThread(gm,shared_data);
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


void KinematicTread::kInitialize(float radius,vector3* pos, vector3* vel, vector3* acc, int n)
{
  cudaSetDevice(streamInfo.device);
  k_n = n;
  k_radius = radius;

  cudaMalloc(&k_pos, k_n*sizeof(vector3));
  cudaMalloc(&k_vel, k_n*sizeof(vector3));
  cudaMalloc(&k_acc, k_n*sizeof(vector3));

  cudaMemcpy(k_pos, pos, k_n*sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(k_vel, vel, k_n*sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(k_acc, acc, k_n*sizeof(vector3), cudaMemcpyHostToDevice);
}

void KinematicTread::doKinematicStep()
{
  cudaSetDevice(streamInfo.device);
  float tolerance = 0.05;
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
  kinematic1stPass<<<1,k_n,0,streamInfo.stream>>>(k_pos, k_n, tolerance, k_radius, k_num_arr);

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

  kinematic2ndPass<<<1,k_n,0,streamInfo.stream>>>(k_pos,k_n,gpu_offset_arr,k_num_arr,tolerance,k_radius,gpu_pair_data);

  cudaMemcpy(cpu_pair_data, gpu_pair_data, contact_sum*sizeof(contactData), cudaMemcpyDeviceToHost);

  //std::cout<<"contact pair num: "<<contact_sum<<std::endl;
  //std::cout<<"i: "<<cpu_pair_data[0].contact_pair.x<<", j:"<<cpu_pair_data[0].contact_pair.y<<std::endl;


  // share data through a common ThreadManager instance
  k_shared_data->contact_pair = cpu_pair_data;
  k_shared_data->contact_pair_n = contact_sum;

  k_shared_data->offset = offset_arr;
  k_shared_data->offset_n = k_n;

  cudaDeviceSynchronize();
}

void DynamicThread::dInitialize(float radius,vector3* pos, vector3* vel, vector3* acc, int n)
{
  cudaSetDevice(streamInfo.device);
  d_n = n;
  d_radius = radius;

  cudaMalloc(&d_pos, d_n*sizeof(vector3));
  cudaMalloc(&d_vel, d_n*sizeof(vector3));
  cudaMalloc(&d_acc, d_n*sizeof(vector3));

  cudaMemcpy(d_pos, pos, d_n*sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vel, vel, d_n*sizeof(vector3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_acc, acc, d_n*sizeof(vector3), cudaMemcpyHostToDevice);
}

void DynamicThread::doDynamicStep()
{
  cudaSetDevice(streamInfo.device);
  //dynamicTestKernel<<<1, 1, 0, dStream>>>();

  // retrieve contact pair data from the ThreadManager
  contactData* cpu_pair = d_shared_data->contact_pair;
  int cpu_pair_n = d_shared_data->contact_pair_n;
  
  int* cpu_offset = d_shared_data->offset;
  int cpu_offset_n = d_shared_data->offset_n;

  // copy data to the Dynamic GPU
  contactData* gpu_pair_data;
  cudaMalloc(&gpu_pair_data, cpu_pair_n*sizeof(contactData));
  cudaMemcpy(gpu_pair_data, cpu_pair, cpu_pair_n*sizeof(contactData), cudaMemcpyHostToDevice);

  // calculate number of threads needed and number of block needed
  int num_thread = 64;
  int num_block = cpu_pair_n / num_thread + 1;

  dynamicPass<<<num_block, num_thread, 0, streamInfo.stream>>>(gpu_pair_data, cpu_pair_n, d_pos, d_vel, d_acc, d_radius);

  cudaDeviceSynchronize();
}