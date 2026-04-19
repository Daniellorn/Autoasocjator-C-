export void ForwardPassISPC(uniform float weights[], uniform float inputs[], uniform float outputs[], uniform int numNeurons, uniform int numInputs)
{
	foreach(i = 0 ... numNeurons)
	{
		float sum = 0.0f;
		for (int j = 0; j < numInputs; j++)
		{
			sum += weights[i * numInputs + j] * inputs[j];
		}
		outputs[i] = 1.0f / (1.0f + exp(-sum));
	}
}