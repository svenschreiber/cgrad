#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

typedef float f32;
typedef int32_t s32;
typedef uint32_t u32;

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

#define MAX_DIMS 8

typedef struct Tensor {
    u32 shape[MAX_DIMS];
    u32 dims;
    f32 *data;
    u32 size;
} Tensor;

typedef struct Linear {
    Tensor w;
    Tensor b;
} Linear;

u32 array_sum(u32 *a, u32 len) {
    u32 sum = 0;
    for (u32 i = 0; i < len; ++i) {
        sum += a[i];
    }
    return sum;
}

u32 array_prod(u32 *a, u32 len) {
    u32 product = 1;
    for (u32 i = 0; i < len; ++i) {
        product *= a[i];
    }
    return product;
}

void print_tensor_axis(Tensor t, u32 axis, u32 *idx) {
    u32 len = t.shape[axis];
    printf("[");
    for (u32 i = 0; i < len; ++i) {
        if (axis < t.dims - 1) {
            print_tensor_axis(t, axis + 1, idx);
        } else {
            printf("%.2f", t.data[(*idx)++]);
        }
        if (i < len - 1) printf(", ");
    }
    printf("]");
}

void print_tensor(Tensor t) {
    if (t.dims == 0) {
        printf("[]\n");
        return;
    }
    u32 idx = 0;
    print_tensor_axis(t, 0, &idx);
    printf("\n");
}

void print_tensor_shape(Tensor t) {
    printf("(");
    for (u32 i = 0; i < t.dims; ++i) {
        printf("%u", t.shape[i]);
        if (i < t.dims - 1) printf(", ");
    }
    printf(")\n");
}

Tensor make_tensor(u32 *shape, u32 dims, f32 *data) {
    Tensor t = {.dims = dims, .data = data, .size = array_prod(shape, dims)};
    memcpy(t.shape, shape, sizeof(u32) * dims);
    return t;
}

Tensor tensor_reshape(Tensor t, u32 *shape, u32 dims) {
    assert(dims <= MAX_DIMS && dims > 0);
    assert(array_prod(shape, dims) == t.size);
    return make_tensor(shape, dims, t.data);
}

Tensor tensor_flatten(Tensor t) {
    u32 shape[1] = {t.size};
    return tensor_reshape(t, shape, 1);
}

f32 random_uniform(f32 low, f32 high) {
    assert(low < high);
    return (high - low) * ((f32)rand() / RAND_MAX) + low;
}

Tensor tensor_uniform(u32 *shape, u32 dims, f32 low, f32 high) {
    u32 size = array_prod(shape, dims);
    f32 *data = (f32 *)malloc(sizeof(f32) * size);
    for (u32 i = 0; i < size; ++i) {
        data[i] = random_uniform(low, high);
    }
    return make_tensor(shape, dims, data);
}

Linear make_linear(u32 in, u32 out) {
    u32 w_shape[] = {out, in};
    u32 b_shape[] = {out};
    f32 bound = 1.f / sqrtf(in);
    return (Linear){.w = tensor_uniform(w_shape, 2, -bound, bound), .b = tensor_uniform(b_shape, 1, -bound, bound)};
}

Tensor tensor_add(Tensor a, Tensor b) {

}

Tensor tensor_mul(Tensor a, Tensor b) {

}

Tensor tensor_linear(Tensor x, Linear l) {

}

int main(int argc, char **argv) {
    //srand(time(0));

    f32 x_data[] = {0.f, 1.f};
    u32 x_shape[] = {2};
    Tensor x = make_tensor(x_shape, ARRAY_LEN(x_shape), x_data);

    Linear l1 = make_linear(2, 2);
    Linear l2 = make_linear(2, 1);

#if 0
    u32 shape[] = {2, 2, 3};
    u32 dims = ARRAY_LEN(shape);
    Tensor t = tensor_uniform(shape, dims, 0.0f, 1.0f);
    print_tensor(t);
    print_tensor_shape(t);

    u32 new_shape[] = {4, 3};
    t = tensor_reshape(t, new_shape, ARRAY_LEN(new_shape));
    print_tensor(t);
    print_tensor_shape(t);

    t = tensor_flatten(t);
    print_tensor(t);
    print_tensor_shape(t);
#endif

    return 0;
}
