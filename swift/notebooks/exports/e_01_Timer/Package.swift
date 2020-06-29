// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "e_01_Timer",
products: [
.library(name: "e_01_Timer", targets: ["e_01_Timer"]),

],
dependencies: [
.package(path: "../e_00_MNISTLoader")
],
targets: [
.target(name: "e_01_Timer", dependencies: ["e_00_MNISTLoader"]),

]
)