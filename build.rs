fn main() {
    println!("cargo:rerun-if-changed=shaders/lomb_scargle.comp");
    let source = std::fs::read_to_string("shaders/lomb_scargle.comp").unwrap();
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(shaderc::TargetEnv::Vulkan, 0);
    // Keep the Float32 source operation order. GPU transcendental implementations
    // may nevertheless differ slightly from the host math library.
    options.set_optimization_level(shaderc::OptimizationLevel::Zero);
    let artifact = compiler
        .compile_into_spirv(
            &source,
            shaderc::ShaderKind::Compute,
            "lomb_scargle.comp",
            "main",
            Some(&options),
        )
        .unwrap();
    std::fs::write(
        std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("lomb_scargle.spv"),
        artifact.as_binary_u8(),
    )
    .unwrap();
}
