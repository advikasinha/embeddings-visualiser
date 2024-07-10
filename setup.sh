echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust to the PATH
echo "Adding Rust to PATH..."
if [[ "$OSTYPE" == "msys" ]]; then
    source $USERPROFILE/.cargo/env
else
    source $HOME/.cargo/env
fi

# Verify Rust installation
echo "Rust version:"
rustc --version
cargo --version