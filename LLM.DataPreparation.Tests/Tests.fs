﻿module LLM.DataPreparation.Tests

open FsUnit

open NUnit.Framework

[<Test>]
let ``build vocabulary`` () =

    // Test
    let vocabulary = DataSource.createVocabulary "Here's some content."

    // Verify
    vocabulary.Count |> should be (greaterThanOrEqualTo 4)

//[<Test>]
//let ``encode something`` () =

//    // Setup
//    let content    = "First of all, some text goes here."
//    let vocabulary = content |> DataSource.createVocabulary
//    let textInput  = "First of all"
    
//    vocabulary 
//    |> Tokenizer.encode textInput
//    |> Array.length
//    |> should be (greaterThan 3)

[<Test>]
let ``get input/target pairs`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let textInput  = "First of all"
    let elementsInRow = 4
    let stride = 1
    let tokens = vocabulary |> Tokenizer.extractTokens textInput

    // Test
    let result = Tokenizer.inputTargetPairs tokens elementsInRow stride

    result |> fst |> Array.length |> should equal 4

//[<Test>]
//let ``decode something`` () =

//    Tokenizer.decode [||]
//    |> should equal false
