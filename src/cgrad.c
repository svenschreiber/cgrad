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
typedef s32 b32;

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

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

b32 tensor_shape_equals(Tensor a, Tensor b) {
    if (a.dims != b.dims) return 0;
    for (u32 i = 0; i < a.dims; ++i) {
        if (a.shape[i] != b.shape[i]) return 0;
    }
    return 1;
}

// TODO: add inplace variants
Tensor tensor_sub(Tensor a, Tensor b) {
    assert(tensor_shape_equals(a, b));
    f32 *data = (f32 *)malloc(sizeof(f32) * a.size);
    for (u32 i = 0; i < a.size; ++i) {
        data[i] = a.data[i] - b.data[i];
    }
    return make_tensor(a.shape, a.dims, data);
}

Tensor tensor_add(Tensor a, Tensor b) {
    assert(tensor_shape_equals(a, b));
    f32 *data = (f32 *)malloc(sizeof(f32) * a.size);
    for (u32 i = 0; i < a.size; ++i) {
        data[i] = a.data[i] + b.data[i];
    }
    return make_tensor(a.shape, a.dims, data);
}

// element-wise multiplication
Tensor tensor_mul(Tensor a, Tensor b) {
    assert(tensor_shape_equals(a, b));
    f32 *data = (f32 *)malloc(sizeof(f32) * a.size);
    for (u32 i = 0; i < a.size; ++i) {
        data[i] = a.data[i] * b.data[i];
    }
    return make_tensor(a.shape, a.dims, data);
}

Tensor tensor_dot(Tensor a, Tensor b) {
    // Only vec * matrix mul for now
    assert(a.dims == 1 && b.dims <= 2 && a.shape[0] == b.shape[1]);

    u32 shape[] = {b.shape[0]};
    f32 *data = (f32 *)calloc(sizeof(f32), shape[0]);

    for (u32 i = 0; i < b.shape[1]; ++i) {
        for (u32 j = 0; j < a.shape[0]; ++j) {
            data[i] += a.data[j] * b.data[i * b.shape[1] + j];
        }
    }

    return make_tensor(shape, 1, data);
}

Tensor tensor_linear(Tensor x, Linear l) {
    return tensor_add(tensor_dot(x, l.w), l.b);
}

f32 relu(f32 x) {
    return max(0.0f, x);
}

f32 sigmoid(f32 x) {
    return 1.0f / (1.0f + expf(-x));
}

f32 sigmoid_prime(f32 x) {
    return x * (1.0f - x);
}

Tensor tensor_func(Tensor x, f32 (*func)(f32)) {
    f32 *data = (f32 *)calloc(sizeof(f32), x.size);
    for (u32 i = 0; i < x.size; ++i) {
        data[i] = func(x.data[i]);
    }
    return make_tensor(x.shape, x.dims, data);
}

Tensor tensor_func_inplace(Tensor x, f32 (*func)(f32)) {
    for (u32 i = 0; i < x.size; ++i) {
        x.data[i] = func(x.data[i]);
    }
    return x;
}

Tensor tensor_scale(Tensor x, f32 s) {
    for (u32 i = 0; i < x.size; ++i) {
        x.data[i] *= s;
    }
    return x;
}

f32 mse_loss(Tensor x, Tensor y) {
    assert(tensor_shape_equals(x, y));
    f32 loss = 0.0f;
    for (u32 i = 0; i < x.size; ++i) {
        loss += (y.data[i] - x.data[i]) * (y.data[i] - x.data[i]);
    }
    return loss / x.size;
}

void print_linear(Linear l) {
    printf("w = ");
    print_tensor(l.w);
    printf("b = ");
    print_tensor(l.b);
}

int main(int argc, char **argv) {
    //srand(time(0));

    f32 x_data[] = {0.f, 1.f};
    u32 x_shape[] = {2};
    Tensor x = make_tensor(x_shape, ARRAY_LEN(x_shape), x_data);

    f32 y_data[] = {1.f};
    u32 y_shape[] = {1};
    Tensor y = make_tensor(y_shape, ARRAY_LEN(y_shape), y_data);

    Linear l1 = make_linear(2, 2);
    Linear l2 = make_linear(2, 1);

    f32 lr = 0.01f;

    printf("x = ");
    print_tensor(x);
    printf("----- l1\n");
    print_linear(l1);
    printf("----- l2\n");
    print_linear(l2);
    printf("-----\n");

    Tensor l1_out = tensor_linear(x, l1);
    printf("x * l1 = ");
    print_tensor(l1_out);

    l1_out = tensor_func_inplace(l1_out, sigmoid);
    printf("sigmoid = ");
    print_tensor(l1_out);

    Tensor l2_out = tensor_linear(l1_out, l2);
    printf("(x * l1) * l2 = ");
    print_tensor(l2_out);

    l2_out = tensor_func_inplace(l2_out, sigmoid);
    printf("sigmoid = ");
    print_tensor(l2_out);

    f32 loss = mse_loss(l2_out, y);
    printf("loss = %f\n", loss);

    Tensor l2_err = tensor_mul(tensor_sub(l2_out, y), tensor_func(l2_out, sigmoid_prime));
    print_tensor(l2_err);
    Tensor l1_err = tensor_mul(tensor_dot(l2_err, l2.w), tensor_func(l1_out, sigmoid_prime));
    print_tensor(l1_err);

    printf("checkpoint\n");
    Tensor l2_dw = tensor_dot(l1_out, l2_err);
    Tensor l1_dw = tensor_dot(x, l1_err);

    l1.w = tensor_sub(l1.w, tensor_scale(l1_dw, lr));
    l2.w = tensor_sub(l2.w, tensor_scale(l2_dw, lr));
    print_tensor(l1.w);
    print_tensor(l2.w);

    return 0;
}
