# Link GitHub .md Docs as Obsidian Directory

https://ukidlucas.blogspot.com/2025/10/obsidian.html

If your Markdown files live in a GitHub repository but you want to edit them directly in Obsidian, you can create a symbolic link.  

This allows Obsidian to treat the remote folder as part of your vault without copying files.

Example (macOS)

> ```bash
> 
> ln -s "/Users/user_name/.../REPO/DNN-book/chapters" "/Users/user_name/Obsidian/DNN-book"