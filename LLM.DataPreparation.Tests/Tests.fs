module LLM.DataPreparation.Tests

open FsUnit
open NUnit.Framework
open LLM.DataPreparation.EmbeddingLayer.WeightMatrix

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
    let vectorSize = 4
    let stride = 1
    let contentTokens = vocabulary |> Tokenizer.extractTokens content

    // Test
    let input, target  = Tokenizer.inputTargetPairs contentTokens vectorSize stride
    let textDictionary = vocabulary |> Tokenizer.toTextDictionary

    let inputText  = input .[0] |> Array.map(fun token -> textDictionary.[token])
    let targetText = target.[0] |> Array.map(fun token -> textDictionary.[token])

    inputText  |> should equal [|"First"; "of"; "all"; ","|]
    targetText |> should equal [| "of"; "all"; ","; "some";|]

[<Test>]
let ``initialize token embedding`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4

    // Test
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions 

    // Verify
    embeddingsDictionary.Count |> should be (greaterThanOrEqualTo 9)


[<Test>]
let ``Retrieve a token embedding`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions 
    let token = 0

    // Test
    let embedding = embeddingsDictionary.[token]

    // Verify
    embedding |> Array.isEmpty |> should equal false

[<Test>]
let ``Add positional encoding to token`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions 
    let token = 3

    // Test
    ()

    // Verify
    Assert.Fail()
    

//[<Test>]
//let ``decode something`` () =

//    Tokenizer.decode [||]
//    |> should equal false
