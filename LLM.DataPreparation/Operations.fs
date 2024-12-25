namespace LLM.DataPreparation

open Language

module Operations =

    module Text =

        type ToTokenizedText = Text -> TokenedText
        type ToTokenMap      = Text -> (TokenedText * Tokens)
        type ToTokens        = Text -> Tokens
        type ToVocabulary    = Text -> Vocabulary
        type ToEmbedding     = Text -> ContentTokens -> Vocabulary -> TokenVectors

    module TokenedText = type ToTokens = TokenedText -> Tokens

    module Token = 
    
        type toText           = Token -> Text
        type ToTokenEmbedding = Token -> EmbeddingsDictionary -> TokenEmbedding
        type ToInputEmbedding = TokenEmbedding -> PositionalEmbedding -> InputEmbedding
        type ToContextVector  = InputEmbedding -> InputEmbeddings     -> ContextVector

    module Tokens =
    
        type ToTokenEmbeddings      = Tokens -> Vocabulary  -> DimensionCount -> TokenEmbeddings
        type ToPositionalEmbeddings = InputTokens -> DimensionCount -> PositionalEmbeddings
        type ToInputEmbeddings      = TokenEmbeddings -> PositionalEmbeddings -> InputEmbeddings

    module Vectors =

        type PredictToken     = TokenVectors   -> Prediction
        type ToInputEmbedding = TokenEmbedding -> PositionalEmbedding -> InputEmbedding

    module Attention =

        module Score =

            type ComputeScore = InputEmbedding -> InputEmbedding -> AttentionScore

        module Weights =

            type Initialize = unit           -> AttentionWeight
            type Normalize  = AttentionScore -> AttentionWeight

        module ContextVector =

            type Compute = InputEmbeddings -> InputAttentionWeights -> ContextVector

    module SelfAttention =

        module WeightParameters = type Initialize = unit -> WeightParameters

        module QueryVector = type Compute = InputEmbedding -> QueryWeightParameters -> QueryVector
        module KeyVector   = type Compute = InputEmbedding -> KeyWeightParameters   -> KeyVector
        module ValueVector = type Compute = InputEmbedding -> ValueWeightParameters -> ValueVector

    module Tokenizer =

        type Encode = Text.ToEmbedding
        type Decode = Vectors.PredictToken

    module DataLoader =

        module Get =

            type InputTargetPair = ContentTokens -> ElementsPerRow -> Stride -> InputTokens[] * TargetTokens[]
            type TokenTensor     = Vocabulary -> Text -> BatchSize -> MaxRowSize -> TokenVectors
    
    module Get =

        open DataLoader

        module Tokens =

            type InputTargetPair = Get.InputTargetPair

        module Vector =

            type Size = TokenVector -> int

        module Vocabulary =

            type Size  = Vocabulary -> int
            type Items = Vocabulary -> (int * TokenedText) seq