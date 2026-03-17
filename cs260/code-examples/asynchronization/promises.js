const delay = (msg, wait) => {
    setTimeout(() => {
        console.log(msg);
    }, 1000 * wait);
};

let x = 0;

new Promise((resolve, reject) => {
    // code executing in promise
    for (let i = 0; i < 3; i++) {
        delay('In promise: ' + x++, i);
    }
});

// code executing AFTER promise initialized
for (let i = 0; i < 3; i++) {
    delay('After promise: ' + x++, i);
}