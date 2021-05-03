const { app, ipcMain } = require('electron')
const { execSync } = require("child_process");

const fs = require('fs')
const fse = require('fs-extra')
var glob = require("glob")
const path = require('path')

const setupPyEnv = () => {
    let envPath = path.join(app.getPath('appData'), 'inscriptio', 'runtime');

    if (!fs.existsSync(envPath)) {
        fs.mkdirSync(envPath)
    }

    glob.sync(path.join(path.dirname(__filename), "inference/!(env)")).map((file) => fse.copySync(file, path.join(envPath, path.basename(file))));

    execSync(path.join(path.dirname(__filename), "inference/setup.sh '")+envPath+"'");
}

ipcMain.on('pyenv.setup', (event, _) => {
    setupPyEnv();
    event.reply('pyenv.setup__reply', 'success');
});


module.exports = { setupPyEnv };
