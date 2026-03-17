const coinToss = new Promise((resolve, reject) => {
    setTimeout(() => {
        if (Math.random() > 0.5) {
            resolve('success!');
        } else {
            reject('error');
        }
    }, 1000);
})

console.log(coinToss);

coinToss
    .then((result) => console.log(`result: ${result}`))
    .catch((err) => console.log(`error: ${err}`))
    .finally(() => console.log(`completed`));

// NOTE:
// if you do not .catch() and the Promise calls reject(), 
// it will throw an error

console.log("hello there");