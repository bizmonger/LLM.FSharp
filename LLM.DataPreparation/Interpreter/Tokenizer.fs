namespace LLM.DataPreparation

open System
open System.Collections.Generic
open LLM.DataPreparation.Operations

module Tokenizer =

    let private getInputTargetPairs : DataLoader.Get.InputTargetPair =
        
        fun vocabulary tokens stride -> (Array.empty,Array.empty)

    let private extractTokens (text: string) (vocabulary: Dictionary<string, int>) : int[] =

        let tokens = text.Split([| ' '; '\t'; '\n'; ','; '.'; ';'; ':' |], StringSplitOptions.RemoveEmptyEntries)

        let tokenValues =
            tokens
            |> Array.choose (fun token ->

                if vocabulary.ContainsKey(token) 
                then Some(vocabulary.[token])
                else None) 

        tokenValues

    let encode : Text.ToEmbedding =

        fun textInput vocabulary ->

            // Step 1: Create input/target pairs (required for training an LLM)
            //-----------------------------------------------------------------
            let tokens = extractTokens textInput vocabulary
            let stride = 1
            let inputTargetPairs = getInputTargetPairs vocabulary tokens stride
            //-----------------------------------------------------------------

            let embeddingsDictionary = Dictionary<int, float[]>()

            // For each entry in the vocabulary
            for KeyValue(key, value) in vocabulary do

                // Generate a float array based on some logic (e.g., using the string length and value)
                // For the sake of this example, we will generate a float[] array based on the value in the inputDict
                // and the length of the input string
                let floatArray = Array.init 3 (fun i -> float (textInput.Length + i))
        
                // Add this array to the embedding dictionary with the value from vocabulary as the key
                embeddingsDictionary.[value] <- floatArray

            //embeddingsDictionary.Values
            Array.empty

    let decode : Vectors.PredictToken =

        fun _ -> "TODO"