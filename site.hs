--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import           Data.Monoid (mappend)
import           Hakyll
import           Hakyll.Web.Pandoc.Biblio
import           Text.Pandoc.Options
import           Control.Monad (liftM)


--------------------------------------------------------------------------------

pandocMathCompiler =
    let mathExtensions = [Ext_tex_math_dollars, Ext_tex_math_double_backslash,
                          Ext_latex_macros, Ext_inline_code_attributes,
                          Ext_footnotes, Ext_literate_haskell]
        defaultExtensions = writerExtensions defaultHakyllWriterOptions
        newExtensions = foldr enableExtension defaultExtensions mathExtensions
        writerOptions = defaultHakyllWriterOptions {
                          writerExtensions = newExtensions,
                          writerHTMLMathMethod = MathJax ""
                        }
    in pandocCompilerWith defaultHakyllReaderOptions writerOptions

-- bibtexCompiler :: String -> String -> Compiler (Item String)
-- bibtexCompiler cslFileName bibFileName = do
--     csl <- load $ fromFilePath cslFileName
--     bib <- load $ fromFilePath bibFileName
--     getResourceBody
--         >>= readPandocBiblio def csl bib
--         >>= return . writePandoc

--------------------------------------------------------------------------------

config :: Configuration
config = defaultConfiguration
    {
        previewPort          = 5000
    }

--------------------------------------------------------------------------------


main :: IO ()
main = hakyllWith config $ do

    -- Move favicon to root
    match "images/favicon.ico" $ do
        route $ constRoute "favicon.ico"
        compile copyFileCompiler

    match "images/**" $ do
        route   idRoute
        compile copyFileCompiler

    match "static/**" $ do
        route   idRoute
        compile copyFileCompiler

--     match "bibliography/*.bib" $ compile $ biblioCompiler
--     match "pages/*.csl" $ compile $ cslCompiler

--     match "papers.markdown" $ do
--         route   $ setExtension "html"
--         compile $ bibtexCompiler
--                   "static/csl/elsevier.csl"
--                   "static/bib/papers.bib"
--             >>= loadAndApplyTemplate "templates/default.html" defaultContext
--             >>= relativizeUrls

    match "css/*" $ do
        route   idRoute
        compile compressCssCompiler

    match (fromList ["about.markdown", "contact.markdown"]) $ do
        route   $ setExtension "html"
        compile $ pandocCompiler
            >>= loadAndApplyTemplate "templates/default.html" defaultContext
            >>= relativizeUrls

--     match "papers.html" $ do
--         route idRoute
--         compile $ do
--             getResourceBody
--                 >>= loadAndApplyTemplate "templates/default.html" defaultContext
--                 >>= relativizeUrls

    match "posts/*" $ do
        route $ setExtension "html"
        compile $ pandocMathCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    create ["archive.html"] $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let archiveCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Archives"            `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/archive.html" archiveCtx
                >>= loadAndApplyTemplate "templates/default.html" archiveCtx
                >>= relativizeUrls


    match "index.html" $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let indexCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Home"                `mappend`
                    defaultContext

            getResourceBody
                >>= applyAsTemplate indexCtx
                >>= loadAndApplyTemplate "templates/default.html" indexCtx
                >>= relativizeUrls

    match "templates/*" $ compile templateCompiler


--------------------------------------------------------------------------------
postCtx :: Context String
postCtx =
    dateField "date" "%B %e, %Y" `mappend`
    defaultContext

