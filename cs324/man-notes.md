# HW 1

## `printf()` and related functions 

- With `write()`, you give the output stream you want to write to, the data you want to write to it, *as well as the number of bytes to write*. 
- `printf()` and `fprintf()` operate on **file streams** (`FILE *`). 
  - They "save up" `write()` calls and send the pending bytes only when it's most efficient to do so.
- `printf()` *always* uses stdout as its file stream, while you specify a file stream for `fprintf()`.
- Instead of explicitly setting the number of bytes to send (like in `write()`), `printf()` and `fprintf()` send bytes until they detect a null byte value (int 0).
- Before calling `write()`, `printf()` and `fprintf()` perform replacements `%` conversions in its string. 
  - e.g. `%s` for strings, `%d` for decimal numbers, `%x` for hex numbers, etc.

## stdin(3)

Every UNIX program (under normal circumstances) has three streams opened for it when it starts up: 

1. One for input, called "standard input". (**stdin**)
2. One for output, called "standard output". (**stdout**)
3. One for printing diagnostic error messages, called "standard error". (**stderr**)

^ These are typicaly attached to the user's terminal. 

stdin, stdout, and stderr have "integer file descriptors" associated with them, each of which has a corresponding preprocessor symbol in `<unistd.h>`:

| Stream | Integer file descriptor | Symbol defined in `<unistd.h>` |  
| ------ | ----------------------- | ------------------------------ |  
| stdin  | 0                       | `STDIN_FILENO`                 |  
| stdout | 1                       | `STDOUT_FILENO`                |  
| stderr | 2                       | `STDERR_FILENO`                |  

(The man page calls stdin, stdout, and sterr "FILEs"...which I'm not sure what they mean by that...)

You can change these file descriptors...but it sounds dangerous.

### Buffering

- **stderr** is **unbuffered**.
- **stdout** is **line-buffered** (when it points to a terminal).
  - Partial lines will not appear until either `fflush` or `exit` are called (man pages for both of those are in section 3), or until a newline is printed.

These can be changed but it sounds like it's complicated and why would I want to do that in the first place.

## memset(3)

### Synopsis

- INCLUDE: `<string.h>`
- SIGNATURE: `void memset(void* s, int c, size_t n)`.
  - `memset()` fills the first `n` bytes of the memory pointed to by `s` w/ the constant byte `c`. 
- RETURNS:
  - `memset()` returns void.

## `exit()`

`exit(n)` terminates the program and returns `n` as the exit status. 

Inside of the `main` function, `exit(n)` and `return n;` do the same thing. (Because the `main` function returns the exit status.)

## open(2)

### Synopsis

- INCLUDE: `<fcntl.h>`
- SIGNATURE: `int open(const char *pathname, int flags);`
- RETURNS:
  - Success: Return new FD (unsigned int).
  - Failure: `-1`.
    - And "<u>errno</u> is set to indicate the error"...whatever that means?

### Flags

Uhhhh I'm not rly sure what the flags are, but an example of one is `O_RDONLY`&mdash;"Read only". For more flags, see the DESCRIPTION section in open(2).

## read(2)

### Synopsis

- INCLUDE: `<unistd.h>`
- SIGNATURE: `size_t read(int fd, void buffer, size_t count);`
    - Attempts to read `count` bytes from the file descriptor `fd` into the buffer starting at `buffer`.
- RETURNS:
  - Successful: Returns the **number of bytes read**.
    - This may be less than `count`&mdash;that's not an error. 
      - e.g. if there are fewer than `count` bytes available for reading. 
      - e.g. if the read is interrupted by a signal.
  - Failure: `-1` (and `errno` is set to indicate the error).

## close(2)

Closes an FD so that it no longer refers to any file. (And after being closed, it may then be reused.)

### Synopsis

- INCLUDE: `<unistd.h>`
- SIGNATURE: `int close(int fd);`
  - `fd` is the file descriptor for the file you're closing.
- RETURNS: 
  - Success: `0`.
  - Failure: `-1` (and `errno` is set to indicate the error).

### Notes

- If the FD closed was the last FD referring to the underlying file description, the resources associated for that fdescription are freed.

## write(2)

I kept forgetting the inputs for `write()` so I'm just going to note them down here.

### Synopsis

- INCLUDE: `<unistd.h>`
- SIGNATURE: `ssize_t write(int fd, const void* buf, size_t count);`
  - Writes up to `count` bytes starting at `buf` into the file associated w/ `fd`.
- RETURNS:
  - Success: Number of bytes written.
  - Failure: `-1` (and `errno` is set).

## getenv(3)

Get the value of an environment variable.

### Synopsis

- INCLUDE: `<stdlib.h>`
- SIGNATURE: `char *getenv(const char *name);`
  - (The input is basically just a string.)
- RETURNS:
  - Success: **Pointer** to value of env var.
  - Failure: `NULL`. 
    - (Fails if inputted name does not match name of any env var.)

# HW 2

I didn't take notes on the terminal commands I was using for HW 2...sorry.

# HW 3

## ps(1)

Displays information about active processes. 

### Modifiers

| Modifier       | Alternative     | Description          | Example |  
| -------------- | --------------- | -------------------- | ------- |  
| `-p (pidlist)` | `p`, `--pid`    | Select by process ID. `(pidlist)` is a single argument that can either be a blank-separated list (in quotes) or a comma-separated list. `-p` can be used multiple times. | `ps -p "1 2" -p 3,4` |  
| `-o (format)`  | `o`, `--format` | Allows you to specify individual output columns. `(format)` is a single arg that may be a blank-separated list (w/ quotes) or a comma-separated list. See STANDARD FORMAT SPECIFIERS in the man page for columns you can print. | `ps -o user,pid,ppid,state,ucmd` |  
| `--forest`     |                 | ASCII art process tree. (That's it... that's all the man page says about this option...)

## fopen(3)

Opens a file and associates a stream with it.

### Synopsis

- INCLUDE: `<stdio.h>`
- SIGNATURE: `FILE *fopen(string pathname, string mode);`  
- RETURNS: 
  - Success: File pointer (`FILE*`). 
  - Failure: `NULL`. (And `errno` is set to indicate error.)

### Parameters

- `pathname` is the path to the file.
- `mode` specifies read/write mode.
  - Options:
    - `r`: Open for **reading**.
    - `r+`: Open for **reading & writing**.
    - `w`: Open for **writing**. Creates file if it doesn't exist. If file does it exist, its length is truncated to zero (i.e. it's erased, I think?).
    - `w+`: Open for **reading & writing**. Creates file if it doesn't exist; truncates the file if it does.
    - `a`: Open for **appending** (adding to the end of file). Creates file if it doesn't exist.
    - `a+`: Open for **reading & appending** (adding to the end of file). Creates file if it doesn't exist.
  - ^ for all except `a` and `a+`, the stream is positioned at the beginning of the file. For `a` and `a+`, the stream is positioned at the end of the file.
- `pathname` and `modes` are not actually strings&mdash;they're C-strings.
  - Specifically, they're `const char *restrict`.
    - (`restrict` is there for compiler optimization. It tells the compiler it can presume certain expectations with the pointer args.)


## fileno(3)

Returns the file descriptor associated with a file stream.

### Synopsis

- INCLUDE: `<stdio.h>`
- SIGNATURE: `int fileno(FILE* stream);`
- RETURN: 
  - Success: FD (unsigned int) associated with `stream`. 
  - Failure: `-1` (and `errno` is set to indicate the error).

### Notes

- For the non-locking counterpart, see unlocked_stdio(3) in the man pages.

## fflush(3)

For output streams: Flushes the stream. (Man page: "forces a write of all user-space buffered data for the given output or update stream, via stream's underlying write function.")

### Synopsis

- INCLUDE: `<stdio.h>`
- SIGNATURE: `int fflush(FILE *stream);`
- RETURNS:
  - Success: `0`.
  - Failure: `EOF` (and `errno` is set to indicate the error).

### Notes

- Any data buffered in a file stream is part of user-spaced memory.
  - Hence, if a parent calls `fork()` when it has data in its buffer, that buffer and all its data will be copied to the child.

## fclose(3)

Flush a stream and close its underlying FD.

### Synopsis

- INCLUDE: `<stdio.h>`
- SIGNATURE: `int fclose(FILE *stream);`
- RETURNS:
  - Success: `0`.
  - Failure: `EOF` (and `errno` is set).
  - ^ In either case, `stream` is unusable. (See [Notes](#fclose3-notes).)

<h3 id="fclose3-notes">Notes</h3>

- Regardless of whether `fclose()` is successful or not, further access to the stream results in undefined behavior (including further `fclose()` calls).

## fork(2)

### Notes

- The child inherits copies of the parent's set of open FDs. Each FD in the child refers to the same open file description as the corresponding description in the parent. 
  - This means that the 2 FDs share open file status flags, file offset, and signal-driven I/O attributes. 
    - (See F_SETOWN and F_SETSIG in fcntl(2)'s man page for more info).

## pipe(2)

### Synopsis

- INCLUDE: `<unistd.h>`
- SIGNATURE: `int pipe(int *pipefd);`
  - `pipefd` is a 2-element array that is filled w/ the pipe's read & write FDs.
- RETURNS:
  - Success: `0`.
  - Error: `-1`.
    - `errno` is set to indicate error.
    - `pipefd` is left unchanged.

### Parameters

Here is what `pipefd` is filled with, if `pipe()` is successful:

- `pipefd[0]`: FD of pipe's **read** end.
- `pipefd[1]`: FD of pipe's **write** end.

## pipe(7)

Overview of pipes and FIFOs (also called "named pipes").

### I/O

- If a process attempts to read from an empty pipe, then read(2) will block until data is available.
- If a process attempts to write to a full pipe, then write(2) blocks until sufficient data has been read from the pipe to allow the write to complete.
- Non-blocking IO is possible by using fcntl(2) to set certain flags. (See man page for more details.)

<br>

- If all FDs referring to the write end have been closed, then reading from the pipe (with read(2)) will see `EOF` (and `read()` will return `0`).
- If all FDs referring to the read end of a pipe have been closed, then writing to the pipe (w/ write(2)) will cause a `SIGPIPE` signal to be generated for the process that made the write call. If that process ignores that signal, then their write call will fail w/ the error `EPIPE`.
- ^ Hence, an application that uses pipe(2) and fork(2) should use suitable close(2) calls to close unnecessary duplicate FDs, thus ensuring that `EOF` and `SIGPIPE`/`EPIPE` are delivered when appropriate.

<br>

- You can't apply lseek(2) to a pipe. (Whatever that is.)

### Pipe Capacity

Pipes have a limited capacity.

### Bidirectionality

On some systems, pipes are bidirectional&mdash;data can be transmitted in both directions btwn pipe ends. This is not possible on Linux, however.

## execve(2)

### Synopsis

- INCLUDE: `<unistd.h>`
- SIGNATURE: `int execve(string pathname, string*[] argv, string*[] envp[]);`
  - `pathname` is a C-string. (Its actual declaration is `const char  *pathname`).
  - `argv` is an array of pointers to strings. (Its actual declaration is `char *const _Nullable argv[]`)
  - `envp` is the same type as `argv`.
- RETURNS:
  - Success: **Nothing** is returned (bc process switches to new program).
  - Failure: returns `-1`, and sets `errno` to indicate error.

### Parameters

- `pathname` is a path to a binary executable you want the process to switch to.
  - C-string.
  - (Can also alternatively lead to an interpreter script ig...see man page for more on that.)
- `argv` becomes the new program's **command-line arguments**.
  - Array of pointers to strings.
  - To follow convention, `argv[0]` should contain the filename associated w/ the file being executed.
  - Must be terminated by a NULL pointer. I.e., `argv[argc]` must equal `NULL`.
- `envp` becomes the new program's environment. (I "environment" as in **environment variables**??)
  - By convention, each string takes the form `key=value`.
  - Must be terminated by a NULL ptr. 

The new program can access `argv` and `envp` by using this signature for `main`:

```c
int main(int argc, char *argv[], char *envp[])
```

## dup(2) - `dup2()`

dup(2) explains three dup functions: `dup()`, `dup2()`, and `dup3()`. These are notes for `dup2()`, which modifies one FD to point to the same open file description as another FD. 

### Synopsis

- INCLUDE: `<unistd.h>`
- SIGNATURE: `int dup2(int ref_fd, int redir_fd);`
  - (Close `redir_fd`, and open it again to point at the same description as `ref_fd`.)
- RETURNS:
  - Success: Returns new FD (`redir_fd`).
  - Error: `-1` (and sets `errno`.)

### Parameters

- In the man pages, `ref_fd` ("reference FD") is called `oldfd` and `redir_fd` ("redirected FD") is caflled `newfd`. I changed the names to make them more descriptive.

### Notes

- If `ref_fd` is not a valid FD, then the call fails, and `redir_fd` is NOT closed.
- If `ref_fd` is a valid FD and `redir_fd` has the same value, then `dup2()` does nothing&mdash;but still returns `redir_fd`.
- If the FD passed in as `redir_fd` was previously opened, it is closed before being reused in `dup2()`.
  - This close is performed silently&mdash;any errors during the close are not reported by `dup2()`.
- The steps of closing and reusing the FD passed in as `redir_fd` are performed atomically.
  - (This avoids race conditions.)

# Lab 1

## pipe(2)

### Synopsis

- INCLUDE: `<unistd.h>`
- SIGNATURE:
  - `int pipe(int pipefd[2])`
    - `pipefd` is a 2-element integer array that is filled with the read/write FDs for the pipe.
      - `pipefd[0]` refers to the pipe's **READ end**.
      - `pipefd[1]` refers to the pipe's **WRITE end**.
- RETURN:
  - Success: `0`.
  - Failure: `-1`, with `errno` set to indicate the error.
    - On failure, `pipefd` is left unchanged.

### Notes

- To build a pipe with flags, use `pipe2()`: `int pipe2(int pipefd[2], int flags)`. 
  - With `flags` = `0`, `pipe2()` has the same effect as `pipe()`.
- pipe(2) actually has examples for building a pipe. That's dope!

## `wait()` - wait(2)

## `waitpid()` - wait(2)

Wait for a state change from a child. "State change" includes:

- Child was terminated.
- Child was stopped by signal.
- Child was resumed by signal.

In the case of a terminated child, wait(2) functions allow the system to reap the child's resources. Without waiting, the terminated child remains a zombie.

### Synopsis

```c
pid_t waitpid(pid_t pid, int *_Nullable wstatus, int options)
```

- INCLUDE: `<sys/wait.h>`
- PARAMETERS:
  - `pid`: 
    - $\text{pid} < -1$: wait for any child process whose **pgid is equal to abs(`pid`)**.
    - $\text{pid} = -1$: wait for **any** child process.
    - $\text{pid} = 0$: wait for any child process who **shares a pgid with the calling process**.
    - $\text{pid} > 0$: wait for the **child whose pid is `pid`**.
  - `wstatus`: Filled w/ information. May be `NULL`.
    - The value of an integer passed in will be filled w/ the child's exit status.
    - (You'll learn more about how to extract other information in Lab 2.)
  - `options`: Flags and stuff. See man page for more details.
- RETURN:
  - Success: **pid of child** whose state changed.
    - (May return `0` if ran with `WNOHANG` and there are child(ren) with `pid` that have not changed state.)
  - Failure: `-1`, and sets `errno`.

### Notes

- I think `pid_t` is just an integer type?

## setpgid(2)

### Synopsis

```c
int setpgid(pid_t pid, pid_t pgid)
```

- INCLUDE: `<unistd.h>`
- PARAMETERS:
  - `pid`: The PID of the **process you're assigning a pgid to**.
  - `pgid`: The **PGID you're giving the process**.
- RETURNS:
  - Success: `0`.
  - Error: `-1`, and `errno` is set.

## exec(3)

## strcmp(3)

