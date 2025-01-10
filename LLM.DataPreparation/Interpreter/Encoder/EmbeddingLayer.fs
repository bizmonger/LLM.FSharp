namespace LLM.DataPreparation.EmbeddingLayer

open System
open System.Collections.Generic
open LLM.DataPreparation.Language
open LLM.DataPreparation.Operations

module WeightMatrix =

    module Vector =

        let private random = System.Random()

        /// Generates a vector of specified size with random weights in a low range.
        /// Parameters:
        /// - `size`: The size of the vector to generate. Must be a positive integer.
        /// - `range`: The maximum absolute value for the random weights. Must be a positive float.
        /// Returns: An array of low random values within the range [-range, range].
        let initializeWeight (size: int) (range: float) : VectorEmbedding =

            if size  <= 0   then raise (ArgumentException("Invalid size: Size must be a positive integer."))
            if range <= 0.0 then raise (ArgumentException("Invalid range: Range must be a positive float."))

            Array.init size (fun _ -> (random.NextDouble() * 2.0 - 1.0) * range)

    module Embeddings =

        let initialize (dimensions:int) (vocabulary:Vocabulary) : EmbeddingsDictionary =

            let range = 3
            let embeddingsDict = Dictionary<int, float[]>()

            for KeyValue(key, value) in vocabulary do

                let floatArray = Vector.initializeWeight dimensions range
                embeddingsDict.[value] <- floatArray

            let sortedDict =
                embeddingsDict
                |> Seq.sortBy (fun kvp -> kvp.Key)
                |> Dictionary

            sortedDict

    module Token =

        let toEmbedding : Token.ToTokenEmbedding =

            fun lookup token -> lookup.[token]
            
    module Tokens =

        let toPositionalEmbeddings : Tokens.ToPositionalEmbeddings =

            fun dimensions input ->

                let addPositions token =

                    let index = Array.IndexOf(input, token)
                    let mutable vector = Array.create dimensions 0.0

                    for i = 0 to dimensions - 1 do
                        vector.[i] <- float(index) + 0.1 + float(i)

                    vector

                let positionalEmbeddings = input |> Array.map addPositions
            
                positionalEmbeddings

        let toInputEmbeddings : Tokens.ToInputEmbeddings =

            fun tokenEmbeddings positionalEmbeddings ->

                let mutable vectors  : float array array = Array.init tokenEmbeddings.Length (fun _ -> [|0.0|])
                let mutable elements : float array = Array.empty

                let count = tokenEmbeddings.Length - 1

                for index = 0 to count do

                    let inputEmbedding = Compute.vectorSum tokenEmbeddings.[index] positionalEmbeddings.[index]
                    vectors.[index] <- inputEmbedding

                vectors

        let toTokenEmbeddings : Tokens.ToTokenEmbeddings =

            fun lookup tokens -> 
            
                tokens |> Array.map(fun t -> t |> Token.toEmbedding lookup)

    module Initialize =

        let matrix (dimensions:int) (embeddingsCount:int) =

            let range = 3
            let dict = Dictionary<int, float[]>()

            for i = 0 to embeddingsCount - 1 do

                let floatArray = Vector.initializeWeight dimensions range
                dict.[i] <- floatArray

            dict