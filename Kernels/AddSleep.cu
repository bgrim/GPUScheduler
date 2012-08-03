#include <stdio.h>

__device__ void addSleep(void *p_us_time)
{ 
    //This method will sleep for clockRate*kernel_time many clock ticks
    // which is equivalent to sleeping for kernel_time milliseconds
    int time = *((int *) p_us_time);

    float AddPerUs = 10.26188; //Ben
//    float AddPerUs = 9.89996; //Scott

    float adds = time*AddPerUs;

    int temp=0;
    while(temp<adds){
       temp++;
    }
}
