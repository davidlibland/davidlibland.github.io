# blog

To edit, pull this repo to a dir called `blog`, and follow the instructions 
[here](https://jaspervdj.be/hakyll/tutorials/github-pages-tutorial.html), adapted below:

```bash
# Temporarily store uncommited changes
git stash

# Verify correct branch
git checkout dev

# Build new files
stack exec blog clean
stack exec blog build

# Get previous files
git fetch --all
git checkout -b master --track origin/master

# Overwrite existing files with new files
cp -a _site/. .

# Commit
git add -A
git commit -m "Publish."

# Push
git push origin master:master

# Restoration
git checkout dev
git branch -D master
git stash pop
```
