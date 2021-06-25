#include "runtime/helpers.h"
EXPORT __attribute__((always_inline)) __int128 rt_decimal_div_10(__int128 val){
   return val/10;
}
EXPORT __attribute__((always_inline)) __int128 rt_decimal_div_100(__int128 val){
   return val/100;
}
EXPORT __attribute__((always_inline)) __int128 rt_decimal_div_1000(__int128 val){
   return val/1000;
}
EXPORT __attribute__((always_inline)) __int128 rt_decimal_div_10000(__int128 val){
   return val/10000;
}
EXPORT __attribute__((always_inline)) __int128 rt_decimal_div_100000(__int128 val){
   return val/100000;
}
EXPORT __attribute__((always_inline)) __int128 rt_decimal_div_1000000(__int128 val){
   return val/1000000;
}
EXPORT __attribute__((always_inline)) __int128 rt_decimal_div_10000000(__int128 val){
   return val/10000000;
}
EXPORT __attribute__((always_inline)) __int128 rt_decimal_div_100000000(__int128 val){
   return val/100000000;
}
