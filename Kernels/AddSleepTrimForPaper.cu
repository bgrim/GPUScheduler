#include <stdio.h>

__device__ void addSleep(void *p_us_time)
{ 
    int time = *((int *) p_us_time);

    float AddPerUs = 9.89759943623274; // GTX-670
    
    float adds = time*AddPerUs;

    int temp=0;
    while(temp<adds){
       temp++;
    }
}
