const coinToss = () => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (Math.random() > 0.5) {
                resolve('success!');
            } else {
                reject('error');
            }
        }, 1000);
    });
};

try {
    const result = await coinToss();
    console.log(`result: ${result}`);
} catch (err) {
    console.error(`error: ${err}`);
} finally {
    console.log(`completed`);
}

console.log("Hello there");