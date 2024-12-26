namespace LLM.DataPreparation.EmbeddingLayer

open LLM.DataPreparation.Operations

module Compute =

    let vectorSum (vectorA:float array) (vectorB:float array) : float array =

        let result = vectorA |> Array.mapi(fun i v -> vectorA.[i] + vectorB.[i])
        result

    let vectorProduct (vectorA:float array) (vectorB:float array) : float array =

        let result = vectorA |> Array.mapi(fun i v -> vectorA.[i] * vectorB.[i])
        result

    let dotProduct (vectorA:float array) (vectorB:float array) : float =

        let result = vectorA |> Array.mapi(fun i v -> vectorA.[i] * vectorB.[i]) |> Array.sum
        result

    let attentionScores : Attention.Scores.Compute =

        fun inputEmbeddings queryVector ->

            let scores = inputEmbeddings |> Array.map (fun embedding -> dotProduct embedding queryVector)
            scores

    let attentionWeights : Attention.Weights.Compute =

        fun scores ->

            let sum = scores |> Array.sum
            let weights = scores |> Array.map(fun s -> s/sum)

            weights