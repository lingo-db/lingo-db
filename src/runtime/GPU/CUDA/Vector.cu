
#include "lingodb/runtime/GPU/CUDA/Vector.cuh"
#include "lingodb/runtime/GPU/CUDA/DeviceHeap.cuh"

namespace cudaRT{
    __device__ Buffer* Vec::allocCurrentCapacity() {
        return reinterpret_cast<Buffer*>(heapAlloc(capacity * sizeof(Buffer)));
    }

    __device__ void Vec::destroy(){
        if(payLoad){
            freeHeapPtr(payLoad);
            payLoad = nullptr;
        }
        capacity = 0;
        numElems = 0;
    }

    __device__ void Vec::grow(){
        capacity *= 2;
        Buffer* newPayLoad = allocCurrentCapacity();
        if (payLoad) {
            for (int32_t i = 0; i < numElems; i++) {
                newPayLoad[i] = payLoad[i];
            }
            freeHeapPtr(payLoad);
        }
        payLoad = newPayLoad;
    }

    __device__ void Vec::push_back(const Buffer& elem){
        if(!payLoad || numElems == capacity){
            grow();
        }
        payLoad[numElems] = elem;
        numElems++;
    }

    __device__ void Vec::merge(Vec* other) {
        if(!other->payLoad) {return;}
        if(this == other) {return;}

        if(payLoad){
            for(int32_t i = 0; i < other->numElems; i++){
                push_back(other->payLoad[i]);
            }
        } else {
            payLoad = other->payLoad;
            capacity = other->capacity;
            numElems = other->numElems;
            other->payLoad = nullptr;
        }
        // other->destroy(); // HUGE PERF IMPACT
    }
}
