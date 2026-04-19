#include <raylib.h>
#include <assert.h>

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <print>
#include <algorithm>

#include "ispcSource/ForwardPass.h"

namespace Math {

    static std::mt19937 engine{ std::random_device{}() };

    float Sigmoid(float s)
    {
        return 1.0f / (1.0f + std::expf(-s));
    }

    float SigmoidDerivative(float s)
    {
        float sig = Sigmoid(s);
        return sig * (1 - sig);
    }

    float RandomFloat(float min, float max)
    {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(engine);
    }

    float RandomInt(int min, int max)
    {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(engine);
    }

    float SigmoidDerivativeFromOutput(float output)
    {
        return output * (1.0f - output);
    }
}


struct WindowSpecification
{
    int width = 1280;
    int height = 720;
};



struct Layer
{
    std::vector<float> weights{};
    std::vector<float> output{};
    std::vector<float> lastInput{};

    std::vector<float> deltas{};
    std::vector<float> weightGradients{};


    int numOfNeurons = 0;
    int numOfInputs = 0;
};

//struct NeuralNet
//{
//    std::vector<Layer> layers;
//    int numOfLayers;
//};

void InitLayer(Layer& layer, int neurons, int inputs)
{   
    layer.numOfInputs = inputs;
    layer.numOfNeurons = neurons;

    layer.weights.resize(neurons * inputs);
    layer.output.resize(neurons);
    layer.deltas.resize(neurons);
    layer.weightGradients.resize(neurons * inputs);
    layer.lastInput.resize(inputs);

    for (int i = 0; i < layer.weights.size(); i++)
    {
        layer.weights[i] = Math::RandomFloat(-0.1f, 0.1f);
    }
}

void ForwardPass(Layer& layer, std::vector<float>& input)
{
    assert(input.size() == layer.numOfInputs);
    layer.lastInput = input;

    //for (int i = 0; i < layer.numOfNeurons; i++)
    //{
    //    float sum = 0.0f;
    //    for (int j = 0; j < layer.numOfInputs; j++)
    //    {
    //        sum += layer.weights[i * layer.numOfInputs + j] * input[j];
    //    }
    //    layer.output[i] = Math::Sigmoid(sum);
    //}

    ispc::ForwardPassISPC(
        layer.weights.data(),       
        input.data(),               
        layer.output.data(),        
        layer.numOfNeurons,         
        layer.numOfInputs           
    );
}

int main(int argc, char** argv)
{
    WindowSpecification spec;
    InitWindow(spec.width, spec.height, "AutoasocjatorXY-RGB");
    
    constexpr int layerCount = 4;
    float learningRate = 0.01f;
   
    Image img = LoadImage("lena_color260x260.png");
    assert(img.data);
    Texture2D tex = LoadTextureFromImage(img);
    
    std::array<Layer, layerCount> neuralNet;
    
    InitLayer(neuralNet[0], 32, 5);
    InitLayer(neuralNet[1], 64, 33);
    InitLayer(neuralNet[2], 64, 65);
    InitLayer(neuralNet[3], 3, 65);

    Image resultImg = GenImageColor(img.width, img.height, BLACK);
    Texture2D resultTex = LoadTextureFromImage(resultImg);
    Color* resultPixels = (Color*)resultImg.data;

    int epochCounter = 0;
    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(DARKGRAY);
        DrawTextureEx(tex, { spec.width / 2.0f - img.width - 300, spec.height / 2.0f - img.height}, 0.0f, 2.0f, WHITE);

        float totalError = 0.0f;
        for (int y = 0; y < img.height; y++)
        {
            for (int x = 0; x < img.width; x++)
            {
                std::vector<float> input = { (float)x / img.width, (float)y / img.height, 1.0f, std::sinf((float)x / img.width), std::sinf((float)y / img.height)};
                auto target = ColorNormalize(GetImageColor(img, x, y));
                const std::array<float, 3> targetArr = { std::clamp(target.x, 0.1f, 0.9f), std::clamp(target.y, 0.1f, 0.9f), std::clamp(target.z, 0.1f, 0.9f) };

                //ForwardPass
                for (int k = 0; k < layerCount; k++)
                {
                    ForwardPass(neuralNet[k], input);
                    input = neuralNet[k].output;
                    input.push_back(1.0f);
                }

                //BackwardPass output layer
                Layer& lastLayer = neuralNet[layerCount - 1];
                for (int l = 0; l < lastLayer.numOfNeurons; l++)
                {
                    float error = lastLayer.output[l] - targetArr[l];
                    totalError += error * error;
                    lastLayer.deltas[l] = error * Math::SigmoidDerivativeFromOutput(lastLayer.output[l]);
                }

                //BackwardPass hidden layer
                for (int m = layerCount - 2; m >= 0; m--)
                {
                    Layer& current = neuralNet[m];
                    Layer& next = neuralNet[m + 1];
                    for (int h = 0; h < current.numOfNeurons; h++)
                    {
                        float errorSum = 0.0f;
                        for (int g = 0; g < next.numOfNeurons; g++)
                        {
                            errorSum += next.deltas[g] * next.weights[g * next.numOfInputs + h];
                        }
                        current.deltas[h] = errorSum * Math::SigmoidDerivativeFromOutput(current.output[h]);
                    }
                }

                //WeightUpdate
                for (int j = 0; j < layerCount; j++)
                {
                    Layer& layer = neuralNet[j];
                    for (int k = 0; k < layer.numOfNeurons; k++)
                    {
                        for (int h = 0; h < layer.numOfInputs; h++)
                        {
                            float grad = layer.deltas[k] * layer.lastInput[h];
                            layer.weights[k * layer.numOfInputs + h] -= learningRate * grad;
                        }
                    }
                }
            }
        }
        epochCounter++;
        for (int y = 0; y < img.height; y++)
        {
            for (int x = 0; x < img.width; x++)
            {
                // 1. Forward Pass
                std::vector<float> input = { (float)x / img.width, (float)y / img.height, 1.0f , std::sinf((float)x / img.width),
            std::sinf((float)y / img.height) };
                for (int k = 0; k < layerCount; k++)
                {
                    ForwardPass(neuralNet[k], input);
                    input = neuralNet[k].output;
                    input.push_back(1.0f);
                }

                // 2. Pobranie wyniku
                resultPixels[y * img.width + x] = {
                                (unsigned char)(neuralNet[layerCount - 1].output[0] * 255),
                                (unsigned char)(neuralNet[layerCount - 1].output[1] * 255),
                                (unsigned char)(neuralNet[layerCount - 1].output[2] * 255),
                                255
                };
            }
        }
        UpdateTexture(resultTex, resultImg.data);
        DrawTextureEx(resultTex, { spec.width / 2.0f - resultImg.width + 300 , spec.height / 2.0f - resultImg.height}, 0.0f, 2.0f, WHITE);
        DrawText(TextFormat("Epoka: %d Czas: %0.2f", epochCounter, GetTime()), 10, 10, 20, BLACK);
        EndDrawing();
    }

    UnloadTexture(resultTex);
    UnloadImage(resultImg);
    CloseWindow();
}
