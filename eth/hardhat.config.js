require("dotenv").config();

require("@nomiclabs/hardhat-etherscan");
require("@nomiclabs/hardhat-waffle");
require("hardhat-gas-reporter");
require("solidity-coverage");
require("@nomiclabs/hardhat-web3");
require("maci-domainobjs");
require("maci-crypto");

const CONTRACT_ADDRESS = "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9";

// This is a sample Hardhat task. To learn how to create your own go to
// https://hardhat.org/guides/create-task.html
task("accounts", "Prints the list of accounts", async (taskArgs, hre) => {
  const accounts = await hre.ethers.getSigners();

  for (const account of accounts) {
    console.log(account.address);
  }
});

task("balance", "Prints an account's balance")
  .addParam("account", "The account's address")
  .setAction(async (taskArgs) => {
    const account = web3.utils.toChecksumAddress(taskArgs.account);
    const balance = await web3.eth.getBalance(account);

    console.log(web3.utils.fromWei(balance, "ether"), "ETH");
  });

task("claim_bounty", "Claim bounty")
  .addParam("publicKey", "bounty issuer's publilckey", '["12394963504092133463590298742771255746910402294421902681602275178368694525156", "2810009863761268199375234926728016029541833696552145042968279544829897552560"]')
  .setAction(async (taskArgs) => {

    const { execSync } = require("child_process");
    const fs = require("fs");
    const snarkjs = require("snarkjs");

    execSync("python3 scripts/quantize_model.py", {
      stdio: "inherit",
    });

    const { Keypair } = require('maci-domainobjs');
    const mimc7 = require('./node_modules/circomlib/src/mimc7.js');
    console.log(Keypair);

    const key1 = new Keypair();
    const pubKey = JSON.parse(taskArgs.publicKey);
    pubKey[0] = BigInt(pubKey[0]);
    pubKey[1] = BigInt(pubKey[1]);

    
    /*
    [
      BigInt("12394963504092133463590298742771255746910402294421902681602275178368694525156"),
      BigInt("2810009863761268199375234926728016029541833696552145042968279544829897552560"),
    ];
    */

    const key2 = new Keypair();
    key2.pubKey.rawPubKey = pubKey;

    const sharedKey = Keypair.genEcdhSharedKey(key1.privKey, key2.pubKey);

    const rawdata = fs.readFileSync('./artifacts/quantization/inputs_ml.json');
    const data = JSON.parse(rawdata);
    console.log(data);

    function tobigint(value) {
      return BigInt(value);
    }

    var to_hash = [];
    var m = 1;
    var p = 10;
    var n = 1;

    var idx = 0;
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < p; j++) {
            to_hash.push(data.X_q[i][j]);
            idx = idx + 1;
            
        }
    }

    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            to_hash.push(data.Yt_q[i][j]);
            idx = idx + 1;
        }
    }

    to_hash.push(data.z_X);
    idx = idx + 1; 
    to_hash.push(data.z_W);
    idx = idx + 1;
    to_hash.push(data.z_b);
    idx = idx + 1;
    to_hash.push(data.z_Y);
    idx = idx + 1;
    to_hash.push(data.sbsY_numerator);
    idx = idx + 1;
    to_hash.push(data.sbsY_denominator);
    idx = idx + 1;
    to_hash.push(data.sXsWsY_numerator);
    idx = idx + 1;
    to_hash.push(data.sXsWsY_denominator);
    idx = idx + 1;

    to_hash.push(data.sYsR_numerator);
    idx = idx + 1;
    to_hash.push(data.sYsR_denominator);
    idx = idx + 1;
    to_hash.push(data.sYtsR_numerator);
    idx = idx + 1;
    to_hash.push(data.sYtsR_denominator);
    idx = idx + 1;
    to_hash.push(data.constant);
    idx = idx + 1;

    to_hash.push(data.z_R);
    idx = idx + 1;
    to_hash.push(data.z_Sq);
    idx = idx + 1;
    to_hash.push(data.sR2sSq_numerator);
    idx = idx + 1;
    to_hash.push(data.sR2sSq_denominator);
    idx = idx + 1;

    const hash_input = mimc7.multiHash(to_hash.map(tobigint), BigInt(0));

    const W_q_enc = data.W_q.map(function(arr) {
      return arr.slice().map(tobigint);
    });

    const b_q_enc = data.b_q.slice().map(tobigint);

    for (let i = 0; i < b_q_enc.length; i++) {
      var val1 = mimc7.multiHash([b_q_enc[i]], BigInt(0));
      var val2 = mimc7.hash(sharedKey, val1);
      b_q_enc[i] = [val1, b_q_enc[i]+val2];
    }

    //console.log(W_q_enc);

    for (let i = 0; i < W_q_enc.length; i++) {
      for (let j = 0; j < W_q_enc[0].length; j++) {
        var val1 = mimc7.multiHash([W_q_enc[i][j]], BigInt(0));
        var val2 = mimc7.hash(sharedKey, val1);
        W_q_enc[i][j] = [val1, W_q_enc[i][j]+val2];
      }
    }
    console.log(b_q_enc);
    //console.log(W_q_enc);

    const _input = {
      hash_input: hash_input,
      private_key: key1.privKey.asCircuitInputs(),
      public_key: key2.pubKey.asCircuitInputs(),
      W_q_enc : W_q_enc,
      b_q_enc : b_q_enc,
    };

    const input = Object.assign({}, data, _input);

    BigInt.prototype.toJSON = function() { return this.toString()  }

    fs.writeFile(
      './artifacts/quantization/inputs.json',
      JSON.stringify(input, null, 2),
      () => {},
    );

    const final_zkey = fs.readFileSync("../circuits/artifacts/lr.zkey");
    const wasm = fs.readFileSync("../circuits/artifacts/lr.wasm");
    const wtns = { type: "mem" };

    const logger = {
        debug: () => { },
        info: console.log,
        warn: console.log,
        error: console.log,
    };

    const verification_key = await snarkjs.zKey.exportVerificationKey(final_zkey);
    await snarkjs.wtns.calculate(input, wasm, wtns, logger);
    const start = Date.now();
    const { proof, publicSignals } = await snarkjs.groth16.prove(final_zkey, wtns, logger);
    console.log("Proof took " + (Date.now() - start) / 1000 + " s");

    const verified = await snarkjs.groth16.verify(verification_key, publicSignals, proof, logger);
    if (!verified) throw new Error("Could not verify the proof");

    const call_data = await snarkjs.groth16.exportSolidityCallData(proof, publicSignals);

    const contract_interface = JSON.parse(fs.readFileSync("artifacts/contracts/libraries/BountyManager.sol/BountyManager.json")).abi;
    var contract = new web3.eth.Contract(contract_interface, CONTRACT_ADDRESS, {
      from: '0x1234567890123456789012345678901234567891', // default from address
      gasPrice: '20000000000' // default gas price in wei, 20 gwei in this case
    });

    //console.log(JSON.parse(call_data));
    contract.methods.collectBounty(call_data).send({from: '0x2546BcD3c84621e976D8185a91A922aE77ECEc30'})
  
    console.log("Your Public Key: ");
    console.log(key1.pubKey.rawPubKey);
    console.log("Your Private Key: ");
    console.log(key1.privKey.rawPrivKey);
    console.log("Success!");
  });

task("deposit_bounty", "Deposit bounty")
  .setAction(async (taskArgs) => {

    const { execSync } = require("child_process");
    const fs = require("fs");

    execSync("python3 scripts/quantize_dataset.py", {
      stdio: "inherit",
    });

    const { Keypair } = require('maci-domainobjs');
    const mimc7 = require('./node_modules/circomlib/src/mimc7.js');
    //console.log(mimc7)

    const key = new Keypair();

    const rawdata = fs.readFileSync('./artifacts/quantization/inputs_dataset.json');
    const data = JSON.parse(rawdata);
    console.log(data);

    function tobigint(value) {
      return BigInt(value);
    }

    var to_hash = [];
    var m = 1;
    var p = 10;
    var n = 1;

    var idx = 0;
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < p; j++) {
            to_hash.push(data.X_q[i][j]);
            idx = idx + 1;
            
        }
    }

    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            to_hash.push(data.Yt_q[i][j]);
            idx = idx + 1;
        }
    }

    to_hash.push(data.z_X);
    idx = idx + 1; 
    to_hash.push(data.z_W);
    idx = idx + 1;
    to_hash.push(data.z_b);
    idx = idx + 1;
    to_hash.push(data.z_Y);
    idx = idx + 1;
    to_hash.push(data.sbsY_numerator);
    idx = idx + 1;
    to_hash.push(data.sbsY_denominator);
    idx = idx + 1;
    to_hash.push(data.sXsWsY_numerator);
    idx = idx + 1;
    to_hash.push(data.sXsWsY_denominator);
    idx = idx + 1;

    to_hash.push(data.sYsR_numerator);
    idx = idx + 1;
    to_hash.push(data.sYsR_denominator);
    idx = idx + 1;
    to_hash.push(data.sYtsR_numerator);
    idx = idx + 1;
    to_hash.push(data.sYtsR_denominator);
    idx = idx + 1;
    to_hash.push(data.constant);
    idx = idx + 1;

    to_hash.push(data.z_R);
    idx = idx + 1;
    to_hash.push(data.z_Sq);
    idx = idx + 1;
    to_hash.push(data.sR2sSq_numerator);
    idx = idx + 1;
    to_hash.push(data.sR2sSq_denominator);
    idx = idx + 1;

    const hash_input = mimc7.multiHash(to_hash.map(tobigint), BigInt(0));

    console.log("Hashed inputs: ");
    console.log(hash_input);
    console.log("Your Public Key: ");
    console.log(key.pubKey.rawPubKey);
    console.log("Your Private Key: ");
    console.log(key.privKey.rawPrivKey);

    console.log("Success!");
  });
// You need to export an object to set up your config
// Go to https://hardhat.org/config/ to learn more

/**
 * @type import('hardhat/config').HardhatUserConfig
 */
module.exports = {
  solidity: "0.6.11",
  networks: {
    hardhat: {
      initialBaseFeePerGas: 0, // workaround from https://github.com/sc-forks/solidity-coverage/issues/652#issuecomment-896330136 . Remove when that issue is closed.
    },
    ropsten: {
      url: process.env.ROPSTEN_URL || "",
      accounts:
        process.env.PRIVATE_KEY !== undefined ? [process.env.PRIVATE_KEY] : [],
    },
  },
  gasReporter: {
    enabled: process.env.REPORT_GAS !== undefined,
    currency: "USD",
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY,
  },
};
