async function cow() {
    return new Promise((resolve) => {
        resolve('moo');
    });
}
const result = cow();
console.log(result);
console.log(await result);