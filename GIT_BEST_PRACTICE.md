# Using Git

- Git is a useful tool for version control - essentially allows the group to work on the project simultaneously, and such that we don't just have folder v1, folder v2, folder v2 final etc!

## Keeping up-to-date
- `git clone https://github.com/THargreaves/NASA-Space-Apps-Challenge/` will create the folder with code/resources inside it
- `git pull origin master` updates any changes from GitHub
	- Aim to use this command every 10-20 mins, especially when we're all coding a lot, otherwise there will be a lot of conflicting code

## Adding your changes
- Check you're on the right branch using `git branch`
- `git add -A` followed by `git commit -a -m "Useful commit message"` followed by `git push origin master` to make your changes automatically online
