#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

typedef struct {
    float *w;
    int wSize;
    float b;
} Neuron;

typedef struct {
    float *in;
    float *out;
    int size;
    int amount;
} Data;

typedef Neuron *Model;

float sigmoidf(float x) {
    return 1.f / (1.f + expf(-x));
}

float getValue(Neuron N, float *in, int size) {
    float result = 0;

    for (int i = 0; i < size; i++) {
        result += N.w[i] * in[i];
    }

    return sigmoidf(result + N.b);
}

float forward_2i_2_1_o(Model M, float *in) {
    float newIn[2] = {getValue(M[0], in, 2), getValue(M[1], in, 2)};
    return getValue(M[2], newIn, 2);
}

float cost(Model M, Data d, float (*forward)(Model, float *)) {
    float result = 0;

    for (int i = 0; i < d.amount; i++) {
        float y = (*forward)(M, &d.in[i * d.size]);
        float diff = y - d.out[i];
        result += diff * diff;
    }

    return result / d.amount;
}

float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

void randModel(Model M, int MSize, int wSize) {
    for (int i = 0; i < MSize; i++) {
        M[i].w = malloc(sizeof(float) * wSize);
        M[i].wSize = wSize;
        for (int j = 0; j < wSize; j++) M[i].w[j] = rand_float();
        M[i].b = rand_float();
    }
}

void freeModel(Model M, int MSize) {
    for (int i = 0; i < MSize; i++) {
        free(M[i].w);
    }
    free(M);
}

void printModel(char *name, Model M, int MSize) {
    printf("Model: %s\n", name);
    for (int i = 0; i < MSize; i++) {
        printf("Neuron %d: [ ", i);
        for (int j = 0; j < M[i].wSize; j++) {
            printf("w%d: %f, ", j, M[i].w[j]);
        }
        printf("b: %f ]\n", M[i].b);
    }
}

#define costM cost(M, d, forward)

void finiteDiff(Model M, Model D, int mSize, float eps, Data d, float (*forward)(Model, float *)) {
    float c = costM;
    float save;

    for (int i = 0; i < mSize; i++) {
        for (int j = 0; j < M[i].wSize; j++) {
            save = M[i].w[j];
            M[i].w[j] += eps;
            D[i].w[j] = (costM - c) / eps;
            M[i].w[j] = save;
        }
        save = M[i].b;
        M[i].b += eps;
        D[i].b = (costM - c) / eps;
        M[i].b = save;
    }
}

void train(Model M, Model D, int mSize, float rate) {
    for (int i = 0; i < mSize; i++) {
        for (int j = 0; j < M[i].wSize; j++) {
            M[i].w[j] -= rate * D[i].w[j];
        }
        M[i].b -= rate * D[i].b;
    }
}

void testModel(Model M, int mSize, Data d, float (*forward)(Model, float *)) {
    for (int i = 0; i < d.amount; i++) {
        printf("Testcase %i: { ", i);
        for (int j = 0; j < d.size; j++) {
            printf("in%i: %f, ", j, d.in[i * d.size + j]);
        }
        printf("expected: %f } Model: %f\n", d.out[i], (*forward)(M, &d.in[i * d.size]));
    }
}

bool getDebug(bool *debug, char *s) {
    if (*s == '\0') return false;
    if (s[1] != '\0') return false;
    switch (*s) {
        case 'y': *debug = true; return true;
        case 'n': *debug = false; return true;
        default : return false;
    }
}

bool getRate(float *rate, char *s) {
    while (*s != '\0' && *s >= '0' && *s <= '9') {
        *rate *= 10;
        *rate += *(s++) - '0';
    }
    if (*(s++) != '.') return false;
    float frac = 0;
    while (*s != '\0' && *s >= '0' && *s <= '9') {
        frac += *(s++) - '0';
        frac /= 10;
    }
    *rate += frac;
    return true;
}

bool getIter(size_t *iter, char *s) {
    while (*s != '\0' && *s >= '0' && *s <= '9') {
        *iter *= 10;
        *iter += *(s++) - '0';
    }
    return (*s == '\0');
}

float gateIn[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

float zeroOut[] = {0, 0, 0, 0};
float andOut[] = {0, 0, 0, 1};
float notOut[] = {0, 1, 0, 1};
float xorOut[] = {0, 1, 1, 0};
float orOut[] = {0, 1, 1, 1};
float nandOut[] = {1, 0, 0, 0};

#define forward forward_2i_2_1_o
#define gateSize 3
#define gateWSize 2
#define eps 1e-1

int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Usage: %s <data> <debug> <rate> <iter>\n", argv[0]);
        return 1;
    }

    bool debug;
    if (!getDebug(&debug, argv[2])) {
        printf("Option %s invalid\n", argv[2]);
        return 3;
    }

    float rate = 0;
    if (!getRate(&rate, argv[3])) {
        printf("Rate %s invalid\n", argv[3]);
        return 4;
    }

    size_t iter = 0;
    if (!getIter(&iter, argv[4])) {
        printf("Iterations %s invalid\n", argv[4]);
        return 5;
    }

    Data data;
    data.in = &gateIn[0][0];
    data.size = 2;
    data.amount = 4;

    switch (argv[1][0]) {
        case '0': data.out = zeroOut; break;
        case '&': data.out = andOut; break;
        case '~': data.out = notOut; break;
        case '^': data.out = xorOut; break;
        case '|': data.out = orOut; break;
        case '>': data.out = nandOut; break;
        default :
            printf("Option %s invalid\n", argv[1]);
            return 2;
    }

    srand(time(NULL));

    Model TestModel = malloc(sizeof(Neuron) * gateSize);
    randModel(TestModel, gateSize, gateWSize);
    Model TmpModel = malloc(sizeof(Neuron) * gateSize);
    randModel(TmpModel, gateSize, gateWSize);

    printModel("Logic Gate", TestModel, gateSize);
    testModel(TestModel, gateSize, data, forward);
    
    for (size_t i = 0; i < iter; i++) {
        finiteDiff(TestModel, TmpModel, gateSize, eps, data, forward);

        if (debug) printf("Cost: %f\n", cost(TestModel, data, forward));

        train(TestModel, TmpModel, gateSize, rate);
    }
    if (debug) printf("Cost: %f\n", cost(TestModel, data, forward));

    printModel("Trained Logic Gate", TestModel, gateSize);
    testModel(TestModel, gateSize, data, forward);

    freeModel(TestModel, gateSize);
    freeModel(TmpModel, gateSize);

    return 0;
}