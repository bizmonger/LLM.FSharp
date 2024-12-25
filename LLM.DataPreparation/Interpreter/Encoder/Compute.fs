namespace LLM.DataPreparation.EmbeddingLayer

open LLM.DataPreparation.Operations

module Compute =

    let attentionScores : Attention.Scores.Compute =

        fun embeddingOfQuery inputEmbeddings ->

            [|[||]|]