const moo1 = new Promise((resolve) => {
    resolve('moo1');
});

moo1
    .then((result) => console.log(result));

console.log("Hello there!");

async function moo2() {
    return new Promise((resolve) => {
        resolve('moo2')
    })
}

console.log(await moo2());
console.log("Hello again!");

const moo3 = async () => {
    return new Promise((resolve) => {
        resolve('moo3');
    });
};

console.log(await moo3());
console.log("Hello once again!");