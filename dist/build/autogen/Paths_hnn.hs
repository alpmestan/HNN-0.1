module Paths_hnn (
    version,
    getBinDir, getLibDir, getDataDir, getLibexecDir,
    getDataFileName
  ) where

import Data.Version (Version(..))
import System.Environment (getEnv)

version :: Version
version = Version {versionBranch = [0,1], versionTags = []}

bindir, libdir, datadir, libexecdir :: FilePath

bindir     = "/home/alp/.cabal/bin"
libdir     = "/home/alp/.cabal/lib/hnn-0.1/ghc-6.10.4"
datadir    = "/home/alp/.cabal/share/hnn-0.1"
libexecdir = "/home/alp/.cabal/libexec"

getBinDir, getLibDir, getDataDir, getLibexecDir :: IO FilePath
getBinDir = catch (getEnv "hnn_bindir") (\_ -> return bindir)
getLibDir = catch (getEnv "hnn_libdir") (\_ -> return libdir)
getDataDir = catch (getEnv "hnn_datadir") (\_ -> return datadir)
getLibexecDir = catch (getEnv "hnn_libexecdir") (\_ -> return libexecdir)

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir ++ "/" ++ name)
