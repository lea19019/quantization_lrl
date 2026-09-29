# Notes

In the project.md we're saying that LRL languages are loosing more than high resource ones, what is exactly that "more"?
Marchisio 2024: a bigger relative drop in benchmark accuracy (mMMLU, MGSM, FLORES) after 4-bit quantization, e.g. non-Latin-script languages lost 1.9% vs 0.7% for Latin-script ones on the 103B model, and human raters saw Japanese drop 16% where the automatic metric showed 1.7%.
Marie & Fujita 2025: a bigger drop in COMET translation score after 4-bit quantization, e.g. French and Japanese lose under 2 points while Bengali and Malayalam lose 8 to 10 and Zulu loses 6.2 on Llama-3.3-70B.
Chimoto 2026: a bigger rise in perplexity on that language's text after 4-bit GPTQ/AWQ with English calibration data, up to 3.5 perplexity points on Llama-3.1-8B, most of which better calibration recovers.


What is the problem? 
Why do we care about this problem? 
What's the potential impact this can have?
What is the solution?
How are we evaluating the solution? 
What are the standard metrics we're using?
Describe our experimentation and evaluation methodology
What is the data we're using?

----------------------

Round to Nearest (RTN)
This algorithm will break the weights of a LLM into groups of N values per row (typically 128 values, and honestly IDK why per row) and given the min and max values of the local group 

