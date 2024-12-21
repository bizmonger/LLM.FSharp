namespace LLM.DataPreparation

open Language

module Operations =

    type ToTokenizeText = Text        -> TokenedText
    type ToTokens       = TokenedText -> Tokens
    type ToTokenMap     = Text        -> (TokenedText * Tokens)
    type Tokenize       = Text        -> Tokens

    type BuildVocabulary = Text -> Vocabulary
    type ToEmbedding     = Text -> Vectors

    module List =

        module Vector =

            type Size     = Vector -> int
            type Elements = Vector -> int

        module Vocabulary =

            type Size  = Vocabulary -> int
            type Items = Vocabulary -> (int * TokenedText) seq
