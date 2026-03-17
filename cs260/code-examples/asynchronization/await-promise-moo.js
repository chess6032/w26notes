const moo = new Promise((resolve) => {
    setTimeout(() => resolve('moo'), 1000);
});
console.log(moo);
console.log(await moo);