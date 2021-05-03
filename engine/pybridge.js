const { app, ipcMain } = require('electron')
const { exec } = require("child_process");

const fs = require('fs')
const fse = require('fs-extra')
var glob = require("glob")
const path = require('path')

const setupPyEnv = () => {
    let envPath = path.join(app.getPath('appData'), 'inscriptio', 'runtime');

    if (!fs.existsSync(envPath)) {
        fs.mkdirSync(envPath)
    }

    glob(path.join(path.dirname(__filename), "inference/!(env)"), (_, file) => file.map(i => fse.copySync(i, path.join(envPath, path.basename(i)))));

    exec(path.join(path.dirname(__filename), "inference/setup.sh '")+envPath+"'", (error, stdout, stderr) => {
        if (error) {
            console.log(`error: ${error.message}`);
            return;
        }
        if (stderr) {
            console.log(`stderr: ${stderr}`);
            return;
        }
        console.log(`stdout: ${stdout}`);
    });

}

ipcMain.on('pyenv.setup', (event, arg) => {
    setupPyEnv();
    event.reply('pyenv.setup-reply', 'success');
});


module.exports = { setupPyEnv };
