// Read the sub files to arrays and merge them into an 
//complete and comply with the requirement for visualization
// Store the final array into a file

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <malloc.h>
#include <sys/stat.h> 
char input_filename[300];
char output_filename[300];
char para_filename[300];
#define __REAL__ double

#define MAX_LINE_LENGTH 100

void readparam(char* parafilename,int* iconf, double* conf)
{
  FILE* file2;
  char  Data[MAX_LINE_LENGTH];
  if((file2=fopen(parafilename,"r")) == NULL)
  { printf("Error opening param file\n");
    return;
  }

  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[0]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[1]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[2]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[3]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[4]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[5]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[6]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[7]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[8]);

  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%le\n",&conf[0]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%le\n",&conf[1]);

  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d\n",&iconf[9]);
  fgets(Data,MAX_LINE_LENGTH,file2);
  fscanf(file2,"%d",&iconf[10]);

  fclose(file2);
}   


void set_pointers (double *** ptr, int dim_y, int dim_z, int size)
{
  int j, k;
  for(k=0; k<dim_y; k++){
    for(j=1;j<dim_z;j++)
      ptr[k][j] = ptr[k][j-1]+size;
    if(k<dim_y-1)
      ptr[k+1][0] =ptr[k][dim_z-1]+size;
  }
}

int main(int argc, const char **argv)
{		
  int dim_x, dim_y, dim_z, frames;
  int size_x, size_y, size_z;
  int sub_len_x, sub_len_y, sub_len_z;
  int valid_len_z,valid_len_y;
  int resolution,save,seed, use_failing;
  int save_binary_file,div_y;
  float T, DT;
  char h[2], x[10], y[10], z[10], sub_x[15], sub_y[15], sub_z[15], time[2], Dtime[4];
  char save_data[10], rand_seed[20], use_failing_str[20];
  char save_bin_str[20], div_y_str[20];
  __REAL__ ***sub_arrays, *final;
  FILE *input_file = NULL;
  FILE *output_file = NULL;
  FILE *para_file;
  int rank;
  int slice;
  int species;
  int iconf[11];
  double conf[2];

  strcpy(para_filename,argv[1]);
  rank=atoi(argv[2]);
  species=atoi(argv[3]);
  readparam(para_filename,iconf, conf);
  resolution  = iconf[0];
  size_x = iconf[1];
  size_y = iconf[2];
  size_z = iconf[3];  
  dim_x = iconf[4];
  dim_y = iconf[5];
  dim_z = iconf[6];  
  save = iconf[7];  
  use_failing = iconf[8];  
  save_binary_file = iconf[9];//save Ca in binary file instead of ascii file 
  div_y= iconf[10];  //block size on y direction for cache
  T  = conf[0];
  DT = conf[1];

  //		slice=atoi(argv[3]);
  //para_filename = "param";

//  para_file = fopen(para_filename,"rb");
//
//  fscanf(para_file, " %s %d %s %d %s %d %s %d %s %d %s %d %s %d %s %f %s %f %s %f %s %f %s %d %s %d", 
//      h, &resolution, x, &size_x, y, &size_y, z, &size_z, sub_x,
//      &dim_x, sub_y, &dim_y, sub_z, &dim_z, save_data, &save, use_failing_str, 
//      &use_failing, time, &T, Dtime, &DT, save_bin_str,&save_binary_file,
//      div_y_str,&div_y);
//  fclose(para_file);
  printf("h %d size_x %d size_y %d size_z %d x_domains %d y_dimention %d "
      "z_dimention %d T %f DT %f use_failing %d input_binary_file %d \n",
      resolution, size_x, size_y, size_z, dim_x, dim_y, dim_z, T, DT, use_failing,save_binary_file);
  frames = (int)(T/DT);
  //frames = 1;
  sub_len_x = (size_x/dim_x)+2;
  sub_len_y = (size_y/dim_y)+2;
  sub_len_z = (size_z/dim_z)+2;

  printf("The total frame is %d \n", frames);
  sub_arrays = (__REAL__***)malloc(dim_y*sizeof(__REAL__**));

  for (int k=0; k<dim_y; k++) 
  {
    sub_arrays[k] = (__REAL__**)malloc(dim_z*sizeof(__REAL__*));
  } 	

  char merge_dir[200];

  //sprintf(outdirname,"merge_dir_%d_%d_%d_%d",h,size_x,size_y,size_z);
  sprintf(merge_dir, "merge_dir_%d_%d_%d_%d_%d", resolution, size_x, size_y, 
      size_z, use_failing);
  int create_dir;
  create_dir = mkdir(merge_dir,0755);
  printf("The create dir is %s \n",merge_dir);
  __REAL__ temp;
  int kk =0;   
  int size_element = sub_len_y*sub_len_z;//size_byte/(sizeof(__REAL__));
  sub_arrays[0][0] = (__REAL__*)malloc(size_element*dim_y*dim_z*sizeof(__REAL__));
  if(sub_arrays[0][0]==NULL)
    printf("Memory alloc failed\n");
  final = (__REAL__*)malloc(size_element*dim_y*dim_z*sizeof(__REAL__));
  set_pointers(sub_arrays, dim_y, dim_z, size_element);
  printf ("The total element in one sub_array is %d %d %d\n",size_element,sub_len_y,sub_len_z );
  for(int k=0; k < frames; k++)
  {
    for(int j =0; j < dim_y; j ++)
    {
      for( int i =0 ; i< dim_z; i ++)
      {
  //read binary file
	if(save_binary_file){
		kk=0;
	  sprintf(input_filename, "output_%d_%d_%d_%d_%d_bin/Ca%d_T%d_rank%d_%d_%d.np",
	      resolution, size_x, size_y, size_z, use_failing, species, k, rank, j, i);
	  input_file = fopen(input_filename, "rb");
	  fseek( input_file, 0, SEEK_SET );
   	kk = fread (sub_arrays[j][i], sizeof(double), sub_len_y*sub_len_z, input_file);
		fclose(input_file);
	}
	else  //read normal file
	{
	  sprintf(input_filename, "output_%d_%d_%d_%d_%d/Ca%d_T%d_rank%d_%d_%d.np",
	      resolution, size_x, size_y, size_z, use_failing, species, k, rank,j, i);
	//printf("Opening '%s'\n", input_filename);
	  input_file = fopen(input_filename, "rb");
	  kk=0;
	  while(!feof(input_file))
		{
	  	fscanf(input_file,"%lf",&sub_arrays[j][i][kk]);
	  	kk++;

		}
	fclose(input_file);
	}
	
      }
    }
    //printf("Read succeed!\n");
    int curr_pos =0;
    int start_z, start_y;

    for(int j =0; j < dim_y; j ++)
    {
      if(j==0||j==(dim_y-1))
      {
	valid_len_y = sub_len_y-1;
	if(j==0&&j==(dim_y-1))
	{
	  valid_len_y = sub_len_y;
	}
      }
      else
      {
	valid_len_y = sub_len_y-2;
      }
      if(j==0)
      {
	start_y=0;
      }
      else
      {
	start_y = 1;
      }

      for(int ii=0; ii< valid_len_y; ii++)
      {
	for( int i =0 ; i< dim_z; i ++)
	{
	  if(i==0||i==(dim_z-1))
	  {
	    valid_len_z = sub_len_z-1;

	    if(i==0&&i==(dim_z-1))
	    {
	      valid_len_z = sub_len_z;
	    }
	  }
	  else
	  {
	    valid_len_z = sub_len_z-2;
	  }
	  if(i==0)
	  {
	    start_z = 0;
	  }
	  else
	  {
	    start_z = 1;
	  }
	  int m;
	  for( m=0; m < valid_len_z; m++)
	  {
	    final[curr_pos] = sub_arrays[j][i][(ii+start_y)*sub_len_z+m+start_z];	
	    curr_pos++;
	  }
	}
      }
    }

    sprintf(output_filename, "%s/Ca%d_T%d_merge.np", merge_dir, species, k);
    printf(" The output file name is %s\n", output_filename);
    output_file = fopen(output_filename,"wb");
    //printf("The total point for frame: %d is: %d\n",k,curr_pos);
    //printf("%le\n",sub_arrays[0][0][1]);
    for(kk=0;kk<curr_pos;kk++)
    {
      if((kk%(sub_len_z*dim_z-dim_z*2+2))==0&&kk>0)
      {
					fprintf(output_file,"\n");
      }
      //else
      //{
      		fprintf(output_file,"%.9e ",final[kk]);
      //}
    }
    fprintf(output_file,"\n");
    //fwrite (final, sizeof(__REAL__), curr_pos, output_file);
    fclose(output_file);
  }		

  free (sub_arrays[0][0]);

  for (int k=0; k<dim_y; k++) 
  {
    free (sub_arrays[k]);
  }

  free(sub_arrays);
  free(final);

  return 0;
}
