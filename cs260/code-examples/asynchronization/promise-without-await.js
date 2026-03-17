const somePromise = new Promise((resolve) => {
    console.log('a promise is made...');
    setTimeout(() => resolve('...and it is kept'), 5000);
});

somePromise
    .then((result) => console.log(`RESULT: ${result}`))
    .catch((err) => console.error(`ERROR: ${err}`))
    .finally(() => console.log('DONE'));

console.log('skibidi rizz');