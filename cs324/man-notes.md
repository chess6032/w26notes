# Introduction

The man notes are helpful but really dense, so I've been keeping these notes as I go through them so that I don't have to dive into them more than once (ideally). 

Each section is the title of the man page that I took notes on. E.g., the "stdin(3)" section is notes on the page accessable via `man 3 stdin` (or `man stdin.3`). A few pages have more than one function, e.g. wait(2).

## Disclaimer

I typically only take notes on the parts relevant to this class/the assignment we're doing, so I can't guarantee that everything you'll need will be in here. (I also can't guarantee everything is accurate.)

## Parameters

- C doesn't have a string data type. If I wrote a parameter as `string`, know that it's actually a char pointer/char array.
- When `_Nullable` is used in a pointer's declaration, that means that you can pass in `NULL` (null pointer) for it.
- I don't write down if a parameter is declared as `const`.
  - If you're getting a compiler error saying smth like, "passing this in discards const qualifiers", check the man page to see if you're violating some const-ness thing.

# TODO:

- exec(3)
- strcmp(3)
- kill
- ummm probably some others too. really I should just go back through each assignment and look at what man pages it recommends reading, and add ones I didn't include the first time.

&nbsp;

- udp(7)
- tcp(7)
- send(2)/sendto(2)
- recv(2)/recvfrom(2)

# man notes

**TABLE OF CONTENTS**

- [`printf()`, `fprintf()`, and `write()`](#printf-fprintf-and-write)
- [stdin(3)](#stdin3)
- [open(2)](#open2)
- [fopen(3)](#fopen3)
- [close(2)](#close2)
- [fclose(3)](#fclose3)
- [fileno(3)](#fileno3)
- [fflush(3)](#fflush3)
- [read(2)](#read2)
- [write(2)](#write2)
- [fork(2)](#fork2)
- [wait(2)](#wait2)
- [setpgid(2)](#setpgid2)
- [pipe(2)](#pipe2)
- [pipe(7)](#pipe7)
- [dup(2)](#dup2)
- [execve(2)](#execve2)
- [getenv(3)](#getenv3)
- [ps(1)](#ps1)
- [memset(3)](#memset3)


## `printf()`, `fprintf()`, and `write()`

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

## open(2)

For opening files to be accessed by FDs.

### `open()`

```c
#include <fcntl.h>

int open(char *pathname, int flags)
```

- DESCRIPTION: Opens file at `pathname`.
- RETURNS:
  - Success: Return new FD (unsigned int).
  - Failure: `-1`, and sets `errno`.

### Flags

Uhhhh I'm not rly sure what the flags are, but an example of one is `O_RDONLY`&mdash;"Read only". For more flags, see the DESCRIPTION section in open(2).


## fopen(3)

For opening a file as a file stream.

<!-- Opens a file and associates a stream with it. -->

### `fopen()`

```c
#include <stdio.h>

FILE* fopen(char *pathname, char *mode)
```

- DESCRIPTION: Opens file at `pathname` and associates a stream with it, which it returns.
- RETURNS: 
  - Success: File stream pointer (`FILE*`). 
  - Failure: `NULL`, and sets `errno`.

### `mode` options

`mode` specifies read/write mode. While it's a string, there are a limited number of options you may pass in for it: 

| Input | Mode                     | Notes |  
| ----- | ------------------------ | ----- |  
| `r`   | **reading**              |       |  
| `r+`  | **reading & writing**    |       |  
| `w`   | **writing**              | Creates file if it doesn't exist. If file does it exist, its length is truncated to zero (i.e. it's erased, I think?). |  
| `w+`  | **reading & writing**    | Creates file if it doesn't exist; truncates the file if it does. |  
| `a`   | **appending**            | Creates file if it doesn't exist. |  
| `a+`  | **reading & appending**  | Creates file if it doesn't exist. |  

- For all except `a` & `a+`, the stream is **positioned at the BEGINNING of the file**. 
- For `a` & `a+`, the stream is positioned at the END of the file.

<!-- - `mode` specifies read/write mode.
  - Options:
    - `r`: Open for **reading**.
    - `r+`: Open for **reading & writing**.
    - `w`: Open for **writing**. Creates file if it doesn't exist. If file does it exist, its length is truncated to zero (i.e. it's erased, I think?).
    - `w+`: Open for **reading & writing**. Creates file if it doesn't exist; truncates the file if it does.
    - `a`: Open for **appending** (adding to the end of file). Creates file if it doesn't exist.
    - `a+`: Open for **reading & appending** (adding to the end of file). Creates file if it doesn't exist.
  - ^ for all except `a` and `a+`, the stream is positioned at the beginning of the file. For `a` and `a+`, the stream is positioned at the end of the file. -->

### Parameter types

Uhhh `pathname` and `mode` are defined w/ type `const char *restrict`...which has to do w/ scope and has something to do w/ compiler optimization??? Idrk.
Just use string literals and I think you'll be fine.

## close(2)

For closing files/FDs.

### `close()`

```c
#include <unistd.h>

int close(int fd)
```

- DESCRIPTION: Closes FD `fd` so that it no longer refers to any file. 
  - (And after being closed, it may then be reused.)
<!--   - `fd` is the file descriptor for the file you're closing. -->
- RETURNS: 
  - Success: `0`.
  - Failure: `-1`, and sets `errno`.

### Notes

- If the FD closed was the last FD referring to the underlying file description, the resources associated for that fdescription are freed.

## fclose(3)

### `fclose()`

```c
#include <stdio.h>

int fclose(FILE *stream)
```
- DESCRIPTION: Flush `stream` and close its underlying FD.
- RETURNS:
  - Success: `0`.
  - Failure: `EOF` (and `errno` is set).
  - ^ In either case, `stream` is unusable. (See [Notes](#fclose3-notes).)

<h3 id="fclose3-notes">Notes</h3>

- Regardless of whether `fclose()` is successful or not, further access to the stream results in undefined behavior (including further `fclose()` calls).


## fileno(3)

File stream &rightarrow; FD.

### `fileno()`

```c
#include <stdio.h>

int fileno(FILE *stream)
```
- DESCRIPTION: Returns the FD associated w/ file stream `stream`.
- RETURNS: 
  - Success: FD (unsigned int) associated with `stream`. 
  - Failure: `-1` (and `errno` is set to indicate the error).

### Notes

- For the non-locking counterpart, see unlocked_stdio(3) in the man pages.

## fflush(3)

For output streams: Flushes the stream. (Man page: "forces a write of all user-space buffered data for the given output or update stream, via stream's underlying write function.")

### `fflush()`

```c
#include <stdio.h>

int fflush(FILE *stream)
```

- DESCRIPTION: Flush `stream`.
  - (Output streams only, I presume?)
- RETURNS:
  - Success: `0`.
  - Failure: `EOF`, and sets `errno`.

### Notes

- Any data buffered in a file stream is part of user-spaced memory.
  - Hence, if a parent calls `fork()` when it has data in its buffer, that buffer and all its data will be copied to the child.


## read(2)

For reading from file streams, accessing them by their FD.

### `read()`

```c
#include <unistd.h>

size_t read(int fd, void *buffer, size_t count)
```

- DESCRIPTION: Attempts to read `count` bytes from the file descriptor `fd` into the buffer starting at `buffer`.
- RETURNS:
  - Successful: Returns the **number of bytes read**.
    - This may be less than `count`&mdash;that's not an error. 
      - e.g. if there are fewer than `count` bytes available for reading. 
      - e.g. if the read is interrupted by a signal.
  - Failure: `-1`, and sets `errno`.

## write(2)

I kept forgetting the inputs for `write()` so I'm just going to note them down here.

### Synopsis

```c
#include <unistd.h>

size_t write(int fd, void *buffer, size_t count)
```

- DESCRIPTION: Writes up to `count` bytes starting at `buffer` into the file associated w/ `fd`.
- RETURNS:
  - Success: Number of bytes written.
  - Failure: `-1`, and sets `errno`.

## fork(2)

### `fork()`

```c
#include <unistd.h>

pid_t fork()
```

- RETURN:
  - Success:
    - PARENT: PID OF CHILD.
    - CHILD: `0`.
  - Failure: `-1`, and sets errno.

### Notes

- The child inherits copies of the parent's set of open FDs. Each FD in the child refers to the same open file description as the corresponding description in the parent. 
  - This means that the 2 FDs share open file status flags, file offset, and signal-driven I/O attributes. 
    - (See F_SETOWN and F_SETSIG in fcntl(2)'s man page for more info).

## wait(2)

Wait for a state change from a child. "State change" includes:

- Child was terminated.
- Child was stopped by signal.
- Child was resumed by signal.

In the case of a terminated child, wait(2) functions allow the system to reap the child's resources. Without waiting, the terminated child remains a zombie.

(I think the only state change we care ab in this class is termination.)

### `wait()`

```c
#include <sys/wait.h>

pid_t wait(int _Nullable *wstatus)
```

- DESCRIPTION: waits for a child to die.
- RETURN:
  - Success: **PID of child** whose state changed.
  - Failure: `-1`, and sets `errno`.

### `waitpid()`

```c
#include <sys/wait.h>

pid_t waitpid(pid_t pid, int _Nullable *wstatus, int options)
```

- DESCRIPTION: like `wait()` but w/ more options.
- RETURN:
  - Success: **pid of child** whose state changed.
    - (May return `0` if ran with `WNOHANG` and there are child(ren) with `pid` that have not changed state.)
  - Failure: `-1`, and sets `errno`.

### Parameters

- `pid`: 
  - $\text{pid} < -1$: wait for any child process whose **pgid is equal to abs(`pid`)**.
  - $\text{pid} = -1$: wait for **any** child process.
  - $\text{pid} = 0$: wait for any child process who **shares a pgid with the calling process**.
  - $\text{pid} > 0$: wait for the **child whose pid is `pid`**.
- `wstatus`: Filled w/ information. May be `NULL`.
  - The value of an integer passed in will be filled w/ the child's exit status.
  - (You'll learn more about how to extract other information in Lab 2.)
- `options`: Flags and stuff. (See below.)
  - Set to `0` if you don't want to add any flags.

### Simplest use

These two statements have the same effect:

- `pid_t pid = waitpid(NULL);`
- `pid_t pid = waitpid(-1, NULL, 0);`

It waits until any child is ready to be terminated and reaps it, and sets `pid` to the reaped child's PID. (Idk what would happen in the case of other state changes.)

### Options (for `waitpid()`)

- `WNOHANG`: Return immediately if no child has exited.
- `WUNTRACED`: Also return if a child has stopped (but not traced via ptrace(2)...or smth).
  - Even if this is not specified, status for traced children which have been stopped is still provided. (I assume provided in `wstatus`? Idk.)
- `WCONTINUED`: Also return if a stopped child has been resumed by delivery of `SIGCONT`.

### Retrieving info (`wstatus`)

If `wstatus` is not `NULL`, then `wait()`/`waitpid()` fill the int it points to w/ status information.

From there, you can use these (function) macros to inspect it. For each of these functions, you'll pass in the *value* `wstatus` points to.

| Macro          | Return type | Return      | Notes |  
| -----          | ----------- | ----------- | ----- |  
| `WIFEXITED`    | bool        | true if **child terminated normally**.                          | "Normal" termination: child was terminated by calling exit(3) or _exit(2), or by returning from `main()`. |  
| `WEXITSTATUS`  | char        | the **exit status of child**.                                   | Should **only use if `WIFEXITED` returned true**. <br/> Return consists of the least sig 8 bits of: the `status` arg that the child specified in a call to exit(3) or _exit(2), or `main()`'s return value. |  
| `WIFSIGNALED`  | bool        | true if the child was ***terminated* by a signal**.             | |  
| `WTERMSIG`     | int?        | the **number of the signal** that cause the child to terminate. | Should **only use if `WIFSIGNALED` returned true**. |  
| `WCOREDUMP`    | bool        | true if the **child produced a core dump**. (See core(5).)      | Should **only use if `WIFSIGNALED` returned true**. <br> (This macro is not available on some UNIX implementations, so you should enclose its use inside `#ifdef WCOREDUMP ... #endif`.) |  
| `WIFSTOPPED`   | bool        | true if the child was ***stopped* by delivery of signal**.      | A child being stopped by a signal is possible only if the wait(2) call was done using `WUNTRACED`, or when the child is being traced (via ptrace(2)). |  
| `WSTOPSIG`     | int?        | the **number of the signal** which caused the child to stop.    | Should **only use if `WIFSTOPPED` returned true.** |  
| `WIFCONTINUED` | bool        | true if the child process was resumed by delivery of `SIGCONT`. | |  

## setpgid(2)

### `setpgid()`

```c
#include <unistd.h>

int setpgid(pid_t pid, pid_t pgid)
```

- DESCRIPTION:
  - `pid`: The PID of the **process you're assigning a pgid to**.
  - `pgid`: The **PGID you're giving the process**.
- RETURNS:
  - Success: `0`.
  - Error: `-1`, and sets `errno`.


## pipe(2)

### `pipe()`

```c
#include <unistd.h>

int pipe(int pipefd[2])
```

- DESCRIPTION: Creates a pipe.
  - `pipefd` is a 2-element array that is filled w/ the pipe's read & write FDs.
- RETURNS:
  - Success: `0`.
  - Error: `-1`, and sets `errno`.
    - `pipefd` is left unchanged.

### Parameters

Here is what `pipefd` is filled with, if `pipe()` is successful:

- `pipefd[0]`: FD of pipe's **read** end.
- `pipefd[1]`: FD of pipe's **write** end.

### Examples

pipe(2) actually has examples for building a pipe. That's dope!

### `pipe2()`

To build a pipe w/ flags, use `pipe2()`:

```c
int pipe2(int pipfd[2], int flags)
```

- With `flags` = `0`, `pipe2()` has the same effect as `pipe()`.


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

## dup(2)

dup(2) explains three dup functions: `dup()`, `dup2()`, and `dup3()`. In CS 324, we've only used `dup2()`, which modifies one FD to point to the same open file description as another FD. 

### `dup2()`

```c
#include <unistd.h>

int dup2(int old_fd, int new_fd)
```

- DESCRPTION: "Give the file associated w/ old FD a new FD to be referenced from."
  - More precisely: Closes `new_fd` and opens it again, this time pointed at the same file description as `old_fd`. 
- RETURNS:
  - Success: Returns new FD (`new_fd`).
  - Error: `-1`, and sets `errno`.

<!-- ### Parameters

- In the man pages, `old_fd` is called `oldfd` and `new_fd` is called `newfd`. I changed the names to try and remember it better...but who knows man. -->

### Notes

- If `old_fd` is not a valid FD, then the call fails, and `new_fd` is NOT closed.
- If `old_fd` is a valid FD and `new_fd` has the same value, then `dup2()` does nothing&mdash;but still returns `new_fd`.
- If the FD passed in as `new_fd` was previously opened, it is closed before being reused in `dup2()`.
  - This close is performed silently&mdash;any errors during the close are not reported by `dup2()`.
- The steps of closing and reusing the FD passed in as `new_fd` are performed atomically.
  - (I don't know what this means. But it avoids race conditions ig.)


## execve(2)

### `execve()`

```c
#include <unistd.h>

int execve(char *pathname, char * _Nullable argv[], char * _Nullable  envp[])
```

- DESCRIPTION:
  - `argv` and `envp` are each an array of string pointers that must be null-terminated.
    - You may pass in `NULL` as your argument for either/both. (`_Nullable`)
- RETURNS:
  - Success: **Nothing** is returned (bc process switches to new program).
  - Failure: `-1`, and sets `errno`.

### Parameters

- `pathname` is a path to a binary executable you want the process to switch to.
  - (Can also alternatively lead to an interpreter script ig...see man page for more on that.)
- `argv` becomes the new program's **command-line arguments**.
  - To follow convention, `argv[0]` should contain the filename associated w/ the file being executed.
  - Must be terminated by a NULL pointer. I.e., `argv[argc]` must equal `NULL`.
- `envp` becomes the new program's environment. (ig "environment" as in **environment variables**??)
  - By convention, each string takes the form `key=value`.
  - Must be terminated by a NULL ptr. 

The new program can access `argv` and `envp` by using this signature for `main`:

```c
int main(int argc, char *argv[], char *envp[])
```
## getenv(3)

For accessing environment variables.

### `getenv()`

```c
#include <stdlib.h> 

char* getenv(char *name)
```

- DESCRIPTION: Returns the value of the environment variable called `name` (if it exists).
- RETURNS:
  - Success: **C-String** of the **env var's value**. (i.e. pointer to the first char in the string.)
  - Failure: `NULL`. 
    - (Fails if `name` does not match the name of any env var.)

## ps(1) 

Shell command for displaying information about active processes. 

### Modifiers

| Modifier       | Alternative     | Description          | Example |  
| -------------- | --------------- | -------------------- | ------- |  
| <code>-p <i>pidlist</i> | `p`, `--pid`    | Select by process ID. *`pidlist`* is a single argument that can either be a blank-separated list (in quotes) or a comma-separated list. `-p` can be used multiple times. | `ps -p "1 2" -p 3,4` |  
| <code>-o <i>format</i></code>  | `o`, `--format` | Allows you to specify individual output columns. *`format`* is a single arg that may be a blank-separated list (w/ quotes) or a comma-separated list. See STANDARD FORMAT SPECIFIERS in the man page for columns you can print. | `ps -o user,pid,ppid,state,ucmd` |  
| `--forest`     |                 | ASCII art process tree. (That's it... that's all the man page says about this option...)

## memset(3)

### `memset()`

```c
#include <string.h>

void memset(void *s, int c, size_t n)
```

- DESCRIPTION: fills the first `n` bytes of the memory pointed to by `s` w/ the constant byte `c`. 
- RETURNS:
  - void.

## udp(7)

USER DATAGRAM PROTOCOL.

- Connectionless.
- Unreliable.
- Per-datagram.

### Description

#### Creating UDP socket

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/udp.h>

udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
```

- When UDP socket is created, its **local & remote addrs are unspecified**.

#### Sending messages w/ UDP

- `sendto()`: Send to destination by passing it in as argument.
- `connect()` &rightarrow; `send()` (or `write()`).
  - `connect()` sets socket's destination.
  - `send()` (or `write()`) sends message.

If you use `connect()`, it is still possible to send to other destinations w/ `sendto()`.

#### Receiving messages w/ UDP

- Socket's **local address can be set via `bind()`**, allowing it to receive messagespackets.
  - (Otherwise, "the socket layer wlil automatically assign a free local port out of the range defined by <u>/proc/sys/net/ipv4/ip_local_port_range/</u> and bind the socket to `INADDR_ANY`"...whatever that means.)
- **All recieve operations return only one packet.**
  - If you read LESS than is present in the packet: Only that much is returned.
  - If you read MORE than is present in the packet: Packet is truncated, and `MSG_TRUNC` flag is set.

`MSG_WAITALL` is not supported...whatever that is.

### Address format

UDP uses the IPv4 `sockaddr_in` address format described in ip(7).

### Other shih

There was some more information that didn't look too important.

## tcp(7)

- Connection-oriented.
- Reliable.
  - Guarantees data arrives in order, and retransmits lost packets.
- Stream-oriented.

### Description

#### Creating TCP socket

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>


tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
```

- When TCP socket is created, its **remote & local addrs are not specified**.

#### Sending messages w/ TCP

- `connect()`: establishes outgoingconnection to another TCP socket.

#### Receiving messages w/ TCP

- `bind()` &rightarrow; `listen()` &rightarrow; `accept()`
  - `bind()` sets the local addr & port.
  - `listen()` makes the socket a "listening socket"&mdash;i.e. a socket factory.
    - (Data cannot be transmitted on listening sockets.)
  - `accept()`: Creates new socket for each incoming connection.

#### Misc

- On individual connections, the socket buffer size must be set prior to `listen()` or `connect()` calls in order to have it take effect. 
  - Maximum sizes for socket buffers are declared via `SO_SNDBUF` and `SO_RCVBUF` mechanisms, but they're limited by values in */proc/sys/net/core/rmem_max* and */proc/sys/net/core/wmem_max* files.
  - (See socket(7) for more info.)
- Send "urgent data" by calling `send()` w/ `MSG_OOB` option. Kernel sends `SIGURG` signal to process that owns the socket receiving this urgent data.

### Address formats

TCP is built on top of IP, so addr formats defined by ip(7) apply to TCP.

TCP supports point-to-point communication only. It does not support broadcasting or multicasting.q

### Other shih

Similar to UDP, the man page for TCP is super long, so I'll just add stuff as I go.

## send(2)

### `send()`

### `sendto()`

## recv(2)

### `recv()`

### `recvfrom()`
