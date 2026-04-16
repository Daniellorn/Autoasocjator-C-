#include <array>
#include <vector>
#include <random>

#include "raylib.h"

namespace Math{
    
    struct Matrix
    {
        float* data = nullptr;
        int rows = 0;
        int cols = 0;
    };    
}

struct Layer
{   
    Math::Matrix weights;
    std::vector<float> output;
    std::vector<float> errors;
    float neurons;
    /*
            |x1, x2, d|
            |x3, x4, e|
        W = |x5, x6, b|
            |.., .., ...|

        Wierszy tyle ile neuronow
    */

    /*
        W1 * input = Z
    */
};

static std::mt19937 engine{std::random_device{}()};

float RandomFloat(float min, float max)
{
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(engine);
}

void InitLayer(Layer& layer, int inputCount, int neuronsCount)
{
    int weightCount = inputCount * neuronsCount;

    layer.weights.data = new float[inputCount * neuronsCount];
    layer.weights.cols = inputCount;
    layer.weights.rows = neuronsCount;
    layer.output.resize(neuronsCount);
    layer.errors.resize(neuronsCount);
    layer.neurons = neuronsCount;


    for (int i = 0; i < weightCount; i++)
    {
        layer.weights.data[i] = RandomFloat(-1.0f, 1.0f);
    }

}

int main(int argc, char** argv)
{
    std::array<Layer, 3> net;

    InitLayer(net[0], 3, 32);
    InitLayer(net[1], 32, 32);
    InitLayer(net[2], 32, 3);

    /*
    InitWindow(1280, 720, "AutoasocjatorXY-RGB");

    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(RED);
        EndDrawing();
    }

    CloseWindow();
    */

}
