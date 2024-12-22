﻿namespace LLM.DataPreparation.EmbeddingLayer

open System
open LLM.DataPreparation.Operations

module WeightMatrix =

    module Vector =

        let private random = System.Random()

        /// Generates a vector of specified size with random weights in a low range.
        /// Parameters:
        /// - `size`: The size of the vector to generate. Must be a positive integer.
        /// - `range`: The maximum absolute value for the random weights. Must be a positive float.
        /// Returns: An array of low random values within the range [-range, range].
        let initializeWeight (size: int) (range: float) : float[] =

            if size <= 0    then raise (ArgumentException("Invalid size: Size must be a positive integer."))
            if range <= 0.0 then raise (ArgumentException("Invalid range: Range must be a positive float."))

            Array.init size (fun _ -> (random.NextDouble() * 2.0 - 1.0) * range)

    module Token =

        let tokenEmbedding : Token.ToTokenEmbedding =

            fun token vocabulary dimensionsCount -> [||]
            
    module Tokens =

        let positionalEmbedding : Tokens.ToPositionalEmbeddings =

            fun input dimensionCount ->

                let addPositions token =

                    let index = Array.IndexOf(input, token)
                    let mutable vector = Array.create dimensionCount 0.0

                    for i = 1 to dimensionCount do
                        vector.[i] <- float(index) + 0.1 + float(i)

                    vector

                let result = input |> Array.map addPositions
            
                result // 1.1, 1.2, 1.3
                       // 2.1, 2.2, 3.3
                       // 3.1, 3.2, 3.3

        let inputEmbedding : Token.ToInputEmbedding =

            fun tokenEmbedding positionalEmbedding -> [||]