const moo = new Promise((resolve) => {
    setTimeout(() => resolve('moo'), 10000);
});
console.log(moo);
console.log(await moo);