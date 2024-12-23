namespace LLM.DataPreparation

open System
open System.Collections.Generic
open System.Text.RegularExpressions
open LLM.DataPreparation.Language
open LLM.DataPreparation.Operations

module Tokenizer =

    let toTextDictionary (inputDict: Dictionary<'K, 'V>) : Dictionary<'V, 'K> =

        let swappedDict = Dictionary<'V, 'K>()
        for kvp in inputDict do
            swappedDict.[kvp.Value] <- kvp.Key

        swappedDict

    let inputTargetPairs : DataLoader.Get.InputTargetPair =
        
        fun contentTokens elementsPerRow stride ->

            let inputResult  = contentTokens |> Array.chunkBySize elementsPerRow
            let TargetResult = contentTokens |> Array.skip stride |> Array.chunkBySize elementsPerRow

            (inputResult,TargetResult)

    let extractTokens (inputString: Text) (vocabulary: Vocabulary) : int[] =
        
        // Step 1: Tokenize the input string using a regular expression that handles partial words and prefixes.
        // Using a regex pattern that is more aligned with subword tokens (if BPE is used).
        let tokens = 
            Regex.Split(inputString, @"(\W+|\s+)") // Modify as needed for BPE token format
            |> Array.filter (fun s -> s.Trim() <> "") // Remove empty tokens

        // Step 2: Match tokens from the vocabulary
        let getTokenValue (token: string) : int option =

            // Try to match the exact token from the dictionary
            if vocabulary.ContainsKey(token) then
                Some(vocabulary.[token])
            else
                // If the exact token doesn't match, check for prefix matches in the dictionary
                let matches = 
                    vocabulary
                    |> Seq.filter (fun kvp -> token.StartsWith(kvp.Key))
                    |> Seq.sortBy (fun kvp -> -kvp.Key.Length) // Sort by the length of the prefix to match the longest first
                    |> Seq.tryHead

                match matches with
                | Some(kvp) -> Some(kvp.Value) // If a match is found, return the corresponding token ID
                | None -> None // No match found, return None

        // Step 3: Extract token values
        let tokenValues =
            tokens
            |> Array.choose (fun token -> getTokenValue token) // Extract token values only for tokens that exist in the dictionary

        // Step 4: Return the token values as an array
        tokenValues

    let encode : Text.ToEmbedding =

        fun textInput contentTokens vocabulary ->

            // Step 1: Create input/target pairs (required for training an LLM)
            //-----------------------------------------------------------------
            let elementsPerRow, stride = 4, 1
            let inputTargetPairs = inputTargetPairs contentTokens elementsPerRow stride
            //-----------------------------------------------------------------

            // Step 2: Convert tokens to embeddings
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
            //-----------------------------------------------------------------

            Array.empty

    let decode : Vectors.PredictToken =

            fun _ -> "TODO"