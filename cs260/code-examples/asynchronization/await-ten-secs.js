const wait10secs = new Promise((resolve) => {
    setTimeout(() => resolve('done'), 10000);
});

console.log(wait10secs);
const result = await wait10secs;
console.log(wait10secs);
console.log(result);
console.log('hello there');
