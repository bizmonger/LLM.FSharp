module LLM.DataPreparation.Tests

open FsUnit
open NUnit.Framework
open LLM.DataPreparation.EmbeddingLayer.WeightMatrix
open LLM.DataPreparation.EmbeddingLayer

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
    let contentTokens = content |> Tokenizer.extractTokens vocabulary

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
[<Ignore("Ignore a test")>]
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

[<Test>]
[<Ignore("Ignore a test")>]
let ``Calculate attention scores`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions

    let textInput   = "some text goes"
    let inputTokens = textInput |> Tokenizer.extractTokens vocabulary
    let secondToken = inputTokens.[1]

    let tokenEmbeddings      = inputTokens     |> Tokens.toTokenEmbeddings embeddingsDictionary
    let positionalEmbeddings = inputTokens     |> Tokens.toPositionalEmbeddings dimensions
    let inputEmbeddings      = tokenEmbeddings |> Tokens.toInputEmbeddings positionalEmbeddings

    let query  = inputEmbeddings.[secondToken]
    
    // Test
    let scores = query |> Compute.attentionScores inputEmbeddings

    // Verify
    Assert.Fail()

[<Test>]
[<Ignore("Ignore a test")>]
let ``Calculate context vector`` () =

    // Setup
    let content    = "First of all, some text goes here."
    let vocabulary = content |> DataSource.createVocabulary
    let dimensions = 4
    let embeddingsDictionary = vocabulary |> Embeddings.initialize dimensions

    let textInput   = "some text goes"
    let inputTokens = textInput |> Tokenizer.extractTokens vocabulary
    let secondToken = inputTokens.[1]

    let tokenEmbeddings      = inputTokens     |> Tokens.toTokenEmbeddings embeddingsDictionary
    let positionalEmbeddings = inputTokens     |> Tokens.toPositionalEmbeddings dimensions
    let inputEmbeddings      = tokenEmbeddings |> Tokens.toInputEmbeddings positionalEmbeddings

    let query  = inputEmbeddings.[secondToken]
    let scores = query |> Compute.attentionScores inputEmbeddings

    // Test
    ()

    // Verify
    Assert.Fail()
    
[<Test>]
let ``Calculate vector product`` () =

    // Setup
    let vectorA = [|1.0;2.0|]
    let vectorB = [|2.0;3.0|]

    // Test
    let vectorProduct = Compute.vectorProduct vectorA vectorB

    // Verify
    vectorProduct.[0] |> should equal 2.0
    vectorProduct.[1] |> should equal 6.0

[<Test>]
let ``Calculate dot product`` () =

    // Setup
    let vectorA = [|1.0;2.0|]
    let vectorB = [|2.0;3.0|]

    // Test
    let dotProduct = Compute.dotProduct vectorA vectorB

    // Verify
    dotProduct |> should equal 8

[<Test>]
let ``Compute vector sum`` () =

    // Setup
    let tokenEmbedding      = [|1.0;1.0;1.0|]
    let positionalEmbedding = [|1.1;1.2;1.3|]

    // Test
    let inputEmbeddings = Compute.vectorSum tokenEmbedding positionalEmbedding

    // Verify
    inputEmbeddings |> should equal [|2.1;2.2;2.3|]

[<Test>]
[<Ignore("Ignore a test")>]
let ``Calculate input embedding`` () =

    // Setup
    let tokenEmbeddings = [|
                           [|1.0;1.0;1.0|]
                           [|1.0;1.0;1.0|]
                         |]

    let positionalEmbeddings = [|
                                [|1.1;1.2;1.3|]
                                [|1.1;1.2;1.3|]
                              |]

    // Test
    let inputEmbeddings = Tokens.toInputEmbeddings tokenEmbeddings positionalEmbeddings

    // Verify
    inputEmbeddings.[0] |> should equal [|2.1;2.2;2.3|]

//[<Test>]
//let ``decode something`` () =

//    Tokenizer.decode [||]
//    |> should equal false
