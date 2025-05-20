
import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn, UpdateDateColumn } from "typeorm";

@Entity("season")
export class Season {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @Column()
  year: number;

  @Column()
  regulation_era: string;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
